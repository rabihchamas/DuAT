import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import jaccard_score, accuracy_score


# Define Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


# Custom dataset class
class ISICDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256)) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256)) / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# Load dataset
def get_dataloader(img_dir, mask_dir, batch_size=8):
    dataset = ISICDataset(img_dir, mask_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    accuracies = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).float()

            preds_np = preds.cpu().numpy().astype(np.uint8)
            masks_np = masks.cpu().numpy().astype(np.uint8)

            for pred, mask in zip(preds_np, masks_np):
                dice_scores.append(dice_coefficient(mask, pred))
                iou_scores.append(jaccard_score(mask.flatten(), pred.flatten()))
                accuracies.append(accuracy_score(mask.flatten(), pred.flatten()))

    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU Score: {np.mean(iou_scores):.4f}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")


# Example usage
if __name__ == "__main__":
    img_dir = "path_to_ISIC2018_images"
    mask_dir = "path_to_ISIC2018_masks"
    model_path = "path_to_your_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device)
    model.to(device)

    dataloader = get_dataloader(img_dir, mask_dir)
    evaluate_model(model, dataloader, device)
