import torch
import numpy as np

def generate_cam(model, image_tensor, target_layer):
    """
    Universal CAM generator using Hooks.
    Works for any model with a GAP + Linear layer structure.
    """
    model.eval()
    features = []
    
    # Hook to capture the feature map from the target layer
    def hook(module, input, output):
        features.append(output.detach())
    
    handle = target_layer.register_forward_hook(hook)
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
    
    handle.remove() # Clean up the hook
    
    # Get the predicted class index
    pred_class = output.argmax(dim=1).item()
    
    # Access weights from the final linear layer (classifier or fc)
    if hasattr(model, 'classifier'):
        fc_weights = model.classifier.weight.data
    else:
        fc_weights = model.fc.weight.data

    # Extract feature maps [Channels, H, W]
    fmap = features[0][0] 
    channels = fmap.shape[0]
    
    # Calculate weighted sum: Class_Weights * Feature_Maps
    class_weights = fc_weights[pred_class]
    cam = class_weights.matmul(fmap.reshape(channels, -1))
    cam = cam.reshape(fmap.shape[1], fmap.shape[2])
    
    # Post-processing: ReLU and 0-1 Normalization
    cam = np.maximum(cam.cpu().numpy(), 0)
    denom = np.max(cam) - np.min(cam) + 1e-8
    return (cam - np.min(cam)) / denom