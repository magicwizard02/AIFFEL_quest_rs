import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import ImageFolderWithXMLBBox  # Custom dataset class

def get_dog_dataloader(image_root, annot_root, batch_size=32, is_train=True, num_workers=4, pin_memory=True):
    """
    Factory function to create a DataLoader for Stanford Dogs with BBox support.
    
    Args:
        image_root (str): Path to reorganized image folder (train or test).
        annot_root (str): Path to original Annotation folder.
        batch_size (int): Number of images per batch.
        is_train (bool): If True, applies training augmentations.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): If True, copies Tensors into device pinned memory.
    """
    
    # 1. Define standard ImageNet transforms
    # Pretrained models like ResNet50 require specific normalization (Mean/Std)
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # Basic augmentation for better generalization
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # 2. Initialize the custom dataset (Includes BBox support for CAM validation)
    dataset = ImageFolderWithXMLBBox(
        root=image_root,
        annot_root=annot_root,
        transform=transform,
        target_size=(224, 224)
    )

    # 3. Create the DataLoader
    # Multi-processing (num_workers) prevents CPU bottlenecks during training
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=is_train, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    return loader