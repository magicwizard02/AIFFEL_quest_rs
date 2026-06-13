import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

def unnormalize(img_tensor):
    """
    Reverses ImageNet normalization.
    Returns a float32 array [0, 1] for better compatibility with calculations.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    
    # [C, H, W] -> [H, W, C]
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * std) + mean
    return np.clip(img, 0, 1)

def calculate_iou(mask_a, mask_b):
    """ Calculates Intersection over Union between two binary masks. """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / (union + 1e-8)

def calculate_iou_at_threshold(heatmap, bbox_raw, img_shape, threshold_ratio):
    img_h, img_w = img_shape[:2]
    # Normalize 0-1
    h_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    h_resized = cv2.resize(h_norm, (img_w, img_h))
    
    # Thresholding
    pred_mask = (h_resized >= threshold_ratio).astype(np.uint8)
    
    # GT Mask
    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    bbox = np.array(bbox_raw).flatten()[:4]
    if np.max(bbox) <= 1.01:
        xmin, ymin, xmax, ymax = int(bbox[0]*img_w), int(bbox[1]*img_h), int(bbox[2]*img_w), int(bbox[3]*img_h)
    else:
        xmin, ymin, xmax, ymax = map(int, bbox)
    cv2.rectangle(gt_mask, (xmin, ymin), (xmax, ymax), 1, -1)
    
    return calculate_iou(pred_mask, gt_mask)


def save_individual_heatmap(heatmap, raw_img, bbox_raw, save_path, title):
    """
    Draws GT (Red) and CAM (Green) boxes and saves.
    raw_img: expects float32 array [0, 1] from unnormalize()
    """
    img_h, img_w = raw_img.shape[:2]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Process Heatmap
    h_resized = cv2.resize(heatmap, (img_w, img_h))
    h_norm = (h_resized - h_resized.min()) / (h_resized.max() - h_resized.min() + 1e-8)
    h_color = cv2.applyColorMap(np.uint8(255 * h_norm), cv2.COLORMAP_JET)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)
    
    # Convert raw_img to uint8 temporarily for cv2.addWeighted
    img_uint8 = np.uint8(255 * raw_img)
    overlay = cv2.addWeighted(img_uint8, 0.6, h_color, 0.4, 0)

    # 2. [FIXED] Ground Truth BBox Logic (Matching your Debug Code)
    bbox = np.array(bbox_raw).flatten()[:4]
    
    # Check if we need to scale normalized coords (0~1) to pixel coords
    if np.max(bbox) <= 1.01:
        xmin_g, ymin_g = int(bbox[0]*img_w), int(bbox[1]*img_h)
        xmax_g, ymax_g = int(bbox[2]*img_w), int(bbox[3]*img_h)
    else:
        xmin_g, ymin_g, xmax_g, ymax_g = map(int, bbox)
    
    # 3. CAM Prediction BBox (Green)
    _, thresh = cv2.threshold(np.uint8(255 * h_norm), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare masks for IoU
    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.rectangle(gt_mask, (xmin_g, ymin_g), (xmax_g, ymax_g), 1, -1)
    
    # Draw Red GT Box on Overlay
    cv2.rectangle(overlay, (xmin_g, ymin_g), (xmax_g, ymax_g), (255, 0, 0), 2)

    iou = 0.0
    if contours:
        best_cnt = max(contours, key=cv2.contourArea)
        xc, yc, wc, hc = cv2.boundingRect(best_cnt)
        # Draw Green Prediction Box
        cv2.rectangle(overlay, (xc, yc), (xc + wc, yc + hc), (0, 255, 0), 2)
        
        pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.rectangle(pred_mask, (xc, yc), (xc + wc, yc + hc), 1, -1)
        iou = calculate_iou(pred_mask, gt_mask)

    # 4. Save Final Image
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title(f"{title} | IoU: {iou:.4f}")
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    return iou

def save_multi_layer_results(img_id, m_name, cam_map, grad_maps_dict, raw_img, bbox_raw):
    m_name = m_name.lower()
    
    # Extract a short prefix (e.g., 'alexnet' -> 'alex', 'resnet50' -> 'res')
    # Or just use the full m_name if you prefer
    prefix = m_name.split('_')[0][:4] 

    # 1. Save CAM
    # Changed "res_" to f"{prefix}_"
    cam_path = f"results/cam/{m_name}/{prefix}_{img_id}.png"
    cam_iou = save_individual_heatmap(cam_map, raw_img, bbox_raw, cam_path, "CAM")

    # 2. Save Grad-CAM layers
    layer_ious = {}
    for layer_name, g_map in grad_maps_dict.items():
        clean_layer = layer_name.replace(".", "_")
        # Changed "res_" to f"{prefix}_"
        g_path = f"results/grad_cam/{m_name}/{clean_layer}/{prefix}_{img_id}.png"
        iou = save_individual_heatmap(g_map, raw_img, bbox_raw, g_path, f"Grad-CAM {layer_name}")
        layer_ious[layer_name] = iou
        
    return cam_iou, layer_ious