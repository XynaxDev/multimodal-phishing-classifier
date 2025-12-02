import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PhishingDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Phishing'],
                yticklabels=['Benign', 'Phishing'])
    plt.title('Confusion Matrix - ResNet50')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def create_model():
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )
    return model


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    class_names = ['Benign', 'Phishing']
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }


def main():
    # Paths
    model_path = r"D:\multimodal-phishing\backend\models\image\best_model.pth"
    test_split = r"D:\multimodal-phishing\runs\image\test_split.json"
    output_dir = r"D:\multimodal-phishing\runs\image"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    if not os.path.exists(test_split):
        print(f"ERROR: Test split not found at {test_split}")
        return
    
    print("="*60)
    print("RESNET50 EVALUATION ON TEST SET")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    with open(test_split, 'r') as f:
        test_data = json.load(f)
    
    test_imgs = test_data['image_paths']
    test_labels = test_data['labels']
    
    print(f"Test set size: {len(test_imgs)} images")
    print(f"  Benign: {test_labels.count(0)}")
    print(f"  Phishing: {test_labels.count(1)}")
    
    # Load model
    print("\nLoading model...")
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"✓ Model loaded from {model_path}")
    
    # Create dataset
    test_dataset = PhishingDataset(test_imgs, test_labels, transform=get_transform())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Evaluate
    print("\n" + "-"*60)
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.2f}%")
    print(f"  Recall:    {results['recall']:.2f}%")
    print(f"  F1 Score:  {results['f1']:.2f}%")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"                Predicted")
    print(f"              Benign  Phishing")
    print(f"  Actual Benign    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"         Phishing  {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    print(f"\nDetailed Classification Report:")
    print(results['classification_report'])
    
    # Save results
    output_file = os.path.join(output_dir, 'test_evaluation.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ Results saved to {output_file}")
    
    # Plot confusion matrix
    try:
        cm_plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(cm, cm_plot_path)
    except Exception as e:
        print(f"Warning: Could not save confusion matrix plot: {e}")
    
    # Per-class accuracy
    tn, fp, fn, tp = cm.ravel()
    benign_accuracy = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
    phishing_accuracy = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    
    print(f"\nPer-Class Accuracy:")
    print(f"  Benign:   {benign_accuracy:.2f}%")
    print(f"  Phishing: {phishing_accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()