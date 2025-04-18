import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pyplot as plt
import logging

class MNISTGradCAM:
    def __init__(self, model, target_layer=None):
        """
        Initialize Grad-CAM with the model and target layer.
        If target_layer is None, automatically find the last Conv/ReLU layer.
        :param model: The trained model.
        :param target_layer: The name of the target layer for Grad-CAM (optional).
        """
        self.model = model
        self.target_layer = target_layer or self._find_last_conv_or_relu_layer()
        self.gradients = None
        self.activations = None

        if not self.target_layer:
            raise ValueError("No suitable Conv or ReLU layer found in the model.")

        logging.info(f"Using target layer: {self.target_layer}")

        # Hook to capture gradients and activations
        self._register_hooks()

    def _find_last_conv_or_relu_layer(self):
        """
        Automatically find the last Conv2d or ReLU layer in the model.
        :return: The name of the last Conv2d or ReLU layer.
        """
        last_layer_name = None
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU)):
                last_layer_name = name
        return last_layer_name

    def _register_hooks(self):
        """
        Register hooks to capture gradients and activations from the target layer.
        """
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._save_activations)
                module.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """
        Save the activations from the forward pass.
        """
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        """
        Save the gradients from the backward pass.
        """
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        """
        Generate the Grad-CAM heatmap for the given input tensor.
        :param input_tensor: The input tensor for which Grad-CAM is generated.
        :return: The Grad-CAM heatmap.
        """
        self.model.eval()
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)  # Apply ReLU to remove negative values

        # Normalize the heatmap
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.detach().cpu().numpy()

    def save(self, cam, output_path):
        """
        Save the Grad-CAM heatmap as an image.
        :param cam: The Grad-CAM heatmap.
        :param output_path: The path to save the image.
        """
        try:
            plt.imshow(cam, cmap='jet', alpha=0.5)
            plt.colorbar()
            plt.savefig(output_path)
            plt.close()
            logging.info(f"Grad-CAM heatmap saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving Grad-CAM heatmap: {e}")

    @staticmethod
    def close_xai():
        """
        Clean up resources if necessary (optional).
        """
        logging.info("Closing Grad-CAM resources.")