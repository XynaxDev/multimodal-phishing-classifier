import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# ==================== CONFIGURATION ====================
DATA_DIR = r"D:\multimodal-phishing\data\images\screenshots"
BENIGN_DIR = os.path.join(DATA_DIR, "benign")
PHISHING_DIR = os.path.join(DATA_DIR, "phishing")

# Training config - OPTIMIZED FOR RESNET50
BATCH_SIZE = 24  # Smaller for stability
LEARNING_RATE = 0.0001  # Much lower
EPOCHS = 25
IMAGE_SIZE = 256
NUM_WORKERS = 2
WEIGHT_DECAY = 0.0001  # Lower regularization

# Early stopping
PATIENCE = 10

# Output directories
MODEL_DIR = r"D:\multimodal-phishing\backend\models\image"
RUN_DIR = r"D:\multimodal-phishing\runs\image"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =======================================================


class PhishingDataset(Dataset):
    """Custom dataset for phishing detection"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def get_transforms(is_train=True):
    """Get data transforms - MODERATE augmentation"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def load_dataset():
    """Load all images and create train/val/test splits"""
    print("Loading dataset...")
    benign_imgs = [os.path.join(BENIGN_DIR, f) for f in os.listdir(BENIGN_DIR) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    phishing_imgs = [os.path.join(PHISHING_DIR, f) for f in os.listdir(PHISHING_DIR) if
                     f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    all_imgs = benign_imgs + phishing_imgs
    all_labels = [0] * len(benign_imgs) + [1] * len(phishing_imgs)

    print(f"Total images: {len(all_imgs)}")
    print(f" Benign: {len(benign_imgs)}")
    print(f" Phishing: {len(phishing_imgs)}")

    # Split: 70% train, 15% val, 15% test
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_imgs, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )

    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"\nSplit sizes:")
    print(f" Train: {len(train_imgs)} images")
    print(f" Val: {len(val_imgs)} images")
    print(f" Test: {len(test_imgs)} images")

    # Save test split
    test_data = {'image_paths': test_imgs, 'labels': test_labels}
    with open(os.path.join(RUN_DIR, 'test_split.json'), 'w') as f:
        json.dump(test_data, f)
    print(f"✓ Test split saved")

    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)


def create_model():
    """Create ResNet50 model - MOST RELIABLE"""
    print("Initializing ResNet50...")
    model = models.resnet50(pretrained=True)

    # Unfreeze last 2 blocks only
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 (last block) and fc
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 2)
    )
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100 * correct / total:.2f}%'})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'confusion_matrix': cm.tolist()
    }


def main():
    # Load data
    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels) = load_dataset()

    # Create model
    model = create_model()
    model = model.to(device)

    # Create datasets
    train_dataset = PhishingDataset(train_imgs, train_labels, transform=get_transforms(True))
    val_dataset = PhishingDataset(val_imgs, val_labels, transform=get_transforms(False))

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Loss and optimizer with class weights
    benign_count = train_labels.count(0)
    phishing_count = train_labels.count(1)
    weights = torch.tensor([1.0, benign_count / phishing_count]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Only optimize unfrozen parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING STARTED - ResNet50 (Transfer Learning)")
    print("=" * 60)

    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val Precision: {val_metrics['precision']:.2f}%, Val Recall: {val_metrics['recall']:.2f}%, Val F1: {val_metrics['f1']:.2f}%")

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metrics'].append(val_metrics)

        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pth'))
            print(f"✓ Best model saved! (Val Acc: {best_val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            break

    # Save training config and history
    config = {
        'model': 'ResNet50',
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'epochs_trained': epoch + 1,
        'image_size': IMAGE_SIZE,
        'train_size': len(train_imgs),
        'val_size': len(val_imgs),
        'test_size': len(test_imgs),
        'best_val_accuracy': best_val_acc
    }

    results = {
        'config': config,
        'history': history
    }

    with open(os.path.join(RUN_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {MODEL_DIR}/best_model.pth")
    print(f"✓ Results saved to: {RUN_DIR}/training_results.json")
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"\nTo evaluate, run: python eval_resnet.py")
    print("=" * 60)


if __name__ == "__main__":
    main()