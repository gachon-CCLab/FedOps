# xai_utils.py
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.linear_model import Ridge
from captum.attr import IntegratedGradients

def apply_gradcam(model, input_tensor, target_layer, target_class=None):
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    targets = None
    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    if len(grayscale_cam.shape) == 2:
        grayscale_cam = np.expand_dims(grayscale_cam, axis=-1)
    
    # Assume input_tensor shape: [1, C, H, W], normalize for visualization
    
    input_image = input_tensor.squeeze().cpu().numpy()
    input_image = input_image.astype(np.float32)
    if input_image.max() > 1.0:
        input_image = input_image / 255.0
    input_image = np.moveaxis(input_image, 0, -1)  # [H, W, C]
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image))
    
    visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def apply_gradcam_configurable(model, input_tensor, label, cfg):
    """
    Apply Grad-CAM using config options.
    cfg.xai:
        enabled: true
        layer: conv2  # or index: 1
        save_path: ./outputs/gradcam.jpg
    """
    model.eval()

    # 自动获取 target_layer（支持字符串名或 index）
    if hasattr(model, cfg.xai.layer):
        target_layer = getattr(model, cfg.xai.layer)
    else:
        # fallback：按 index 查找 nn.Conv2d 层
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        try:
            target_layer = conv_layers[int(cfg.xai.layer)]
        except:
            raise ValueError(f"cfg.xai.layer: {cfg.xai.layer} 无效。应为模型属性名或 Conv2d 索引。")

    # 设置目标类
    targets = [ClassifierOutputTarget(int(label))]
    input_image = input_tensor.squeeze().cpu().numpy()   # shape: (C, H, W) or (H, W)
    if input_image.ndim == 2:
        input_image = np.expand_dims(input_image, axis=0)  # → (1, H, W)

    # 将灰度图复制成 RGB (1通道 → 3通道)
    if input_image.shape[0] == 1:
        input_image = np.repeat(input_image, 3, axis=0)  # → (3, H, W)

    input_image = np.moveaxis(input_image, 0, -1).astype(np.float32)  # → (H, W, 3)

    # 归一化到 [0,1]
    if input_image.max() > 1.0:
        input_image /= 255.0
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image) + 1e-8)
    # 初始化 GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # shape: (H, W)


    # 可视化
    vis = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

    # 保存图像
    if hasattr(cfg.xai, "save_path") and cfg.xai.save_path:
        save_path = cfg.xai.save_path
    else:
        save_path = f"./outputs/gradcam_class_{label}.jpg"

    cv2.imwrite(save_path, vis[:, :, ::-1])  # BGR
    print(f"[XAI] Grad-CAM saved to: {save_path}")

    return vis, grayscale_cam


def grad_cam_lstm(model, x):
    """
    Return CAM: shape [T] — 每个时间点的重要性
    """
    model.eval()
    T = x.shape[1]

    # ---- Hook LSTM3 hidden output ----
    activations = []
    gradients   = []

    def forward_hook(module, input, output):
        activations.append(output[0].detach())  # [B,T,H]

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())  # [B,T,H]

    h = model.lstm3.register_forward_hook(forward_hook)
    g = model.lstm3.register_backward_hook(backward_hook)

    logit = model(x)  # forward
    model.zero_grad()
    logit.backward()

    h.remove(); g.remove()

    A = activations[0].squeeze(0)   # [T,H]
    G = gradients[0].squeeze(0)     # [T,H]

    # ---- CAM(t) = ReLU(sum(G_t * A_t)) ----
    cam_t = torch.relu((G * A).sum(dim=1))  # [T]

    # normalize
    cam_t = cam_t / (cam_t.max() + 1e-6)
    return cam_t.cpu().numpy()



def ig_ts(model, x):
    model.eval()
    ig = IntegratedGradients(model)

    x.requires_grad_(True)
    attr = ig.attribute(x, target=None)
    attr = attr.squeeze(0).detach().numpy()  # [T,F]

    # normalize
    attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-9)
    return attr

def lime_ts(model, x_np, num_samples=200):
    """
    x_np: [T,F]
    返回 importance: shape [T,F]
    """
    T, F = x_np.shape
    x_base = x_np.copy()
    importance = np.zeros((T, F))

    with torch.no_grad():
        base_logit = model(torch.tensor(x_base, dtype=torch.float32).unsqueeze(0)).item()

    for t in range(T):
        for f in range(F):
            scores = []
            for _ in range(num_samples):
                x_perturb = x_base.copy()
                x_perturb[t, f] = 0  # mask this feature at this time
                with torch.no_grad():
                    logit = model(torch.tensor(x_perturb, dtype=torch.float32).unsqueeze(0)).item()
                scores.append(base_logit - logit)
            importance[t, f] = np.mean(scores)

    # normalize
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)
    return importance
