import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def apply_gradcam_configurable(model, input_tensor, label, cfg):
    """
    Apply Grad-CAM using configurable settings from cfg.xai.
    Expected cfg.xai fields:
        enabled: true / false
        layer: 'conv2' or numeric index of Conv2d layer
        save_path: output file path for Grad-CAM image
    """
    model.eval()

    # Automatically get the target layer (supports name or index)
    if hasattr(model, cfg.xai.layer):
        target_layer = getattr(model, cfg.xai.layer)
    else:
        # Fallback: find Conv2d layer by index
        conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        try:
            target_layer = conv_layers[int(cfg.xai.layer)]
        except:
            raise ValueError(f"cfg.xai.layer: {cfg.xai.layer} is invalid. It should be an attribute name or Conv2d index.")

    # Set the target class
    targets = [ClassifierOutputTarget(int(label))]
    
    # Convert input tensor to numpy image (H, W, 3)
    input_image = input_tensor.squeeze().cpu().numpy()  # (C, H, W) or (H, W)
    if input_image.ndim == 2:
        input_image = np.expand_dims(input_image, axis=0)  # → (1, H, W)

    # Duplicate grayscale to RGB if necessary
    if input_image.shape[0] == 1:
        input_image = np.repeat(input_image, 3, axis=0)  # → (3, H, W)

    input_image = np.moveaxis(input_image, 0, -1).astype(np.float32)  # → (H, W, 3)

    # Normalize to [0,1]
    if input_image.max() > 1.0:
        input_image /= 255.0
    input_image = (input_image - np.min(input_image)) / (np.max(input_image) - np.min(input_image) + 1e-8)

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # shape: (H, W)

    # Overlay Grad-CAM heatmap
    vis = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

    # Determine save path
    if hasattr(cfg.xai, "save_path") and cfg.xai.save_path:
        save_path = cfg.xai.save_path
    else:
        save_path = f"./outputs/gradcam_class_{label}.jpg"

    # Save as BGR image (for OpenCV)
    cv2.imwrite(save_path, vis[:, :, ::-1])
    print(f"[XAI] Grad-CAM saved to: {save_path}")

    return vis, grayscale_cam
