# explainers/acol.py

import torch
import torch.nn.functional as F
import numpy as np

class ACOLInterpreter:
    def __init__(self, model):
        """
        ACOL (Adversarial Complementary Learning) Interpreter.
        Expects a model with two branches (e.g., A and B).
        """
        self.model = model

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        # ACOL models typically output two sets of feature maps: 
        # one from the primary branch and one from the adversarial branch.
        with torch.no_grad():
            # Assuming the model forward returns (output, map_A, map_B)
            output, map_A, map_B = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Combine the two maps to get the 'Full' object extent
        # map_A: Most discriminative part (e.g., face)
        # map_B: Complementary parts (e.g., body, tail)
        combined_map = torch.max(map_A[:, target_class], map_B[:, target_class])
        
        # Normalization
        combined_map = combined_map.squeeze().cpu().numpy()
        combined_map = (combined_map - np.min(combined_map)) / (np.max(combined_map) + 1e-8)
        
        return combined_map

    def get_adversarial_mask(self, map_A, threshold=0.5):
        """
        Generates a binary mask from the primary map to 'erase' 
        the most discriminative regions for the adversarial branch.
        """
        mask = torch.ones_like(map_A)
        mask[map_A > threshold] = 0.0
        return mask