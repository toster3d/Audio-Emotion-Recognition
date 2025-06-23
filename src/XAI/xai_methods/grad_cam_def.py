import os

import matplotlib.pyplot as plt
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Rejestrujemy hooki
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        """
        input_tensor: torch.Tensor [1, C, H, W]
        class_idx: int - indeks klasy (jeśli None, używa klasy z najwyższym wynikiem)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        loss = output[:, class_idx]
        loss.backward()

        # Średnie gradienty dla każdego kanału feature map
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Ważone sumowanie feature map
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU + normalizacja
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )

        # Normalizacja do [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze(0).cpu().numpy()  # [1, H, W] -> [H, W]

    def visualize(self, cam, original_image, save_path=None):
        plt.figure(figsize=(8, 6))
        plt.imshow(original_image.squeeze(), cmap="gray")  # oryginalny spektrogram
        plt.imshow(cam.squeeze(), cmap="jet", alpha=0.5)  # nałożony Grad-CAM
        plt.axis("off")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()
            print(f"Saved Grad-CAM to {save_path}")
        else:
            plt.show()
