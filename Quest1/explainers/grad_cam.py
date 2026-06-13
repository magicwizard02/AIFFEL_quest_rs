import torch
import numpy as np

class GradCAM:
    """
    Grad-CAM class to capture activations and gradients for any specific layer.
    """
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._hook_layers()

    def _hook_layers(self):
        # Forward hook to save activations
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        # Backward hook to save gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find the target layer by name and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Removes all registered hooks to free up memory."""
        for hook in self.hooks:
            hook.remove()

    def generate(self, input_tensor, target_class=None):
        """Generates a Grad-CAM heatmap."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for the target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None:
            raise ValueError(f"Gradients not captured for: {self.target_layer_name}")

        # Global Average Pooling of gradients (the 'alpha' weights)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        grad_cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU and Normalization
        grad_cam = torch.relu(grad_cam).cpu().numpy()
        denom = np.max(grad_cam) - np.min(grad_cam) + 1e-8
        return (grad_cam - np.min(grad_cam)) / denom