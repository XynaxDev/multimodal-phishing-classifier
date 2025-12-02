import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
MODEL_PATH = r"D:\multimodal-phishing\backend\models\text\bert_finetuned"
DATA_PATH = r"D:\multimodal-phishing\data\text\combined_reduced_splits.csv"
OUTPUT_DIR = r"D:\multimodal-phishing\runs\text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    """Load fine-tuned BERT model"""
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {MODEL_PATH}")
    return tokenizer, model


def load_test_data():
    """Load test split from CSV"""
    print("\nLoading test data from CSV...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data not found at {DATA_PATH}")
        return None, None
    
    df = pd.read_csv(DATA_PATH)
    test_df = df[df['split'] == 'test'].copy()
    
    if len(test_df) == 0:
        print("ERROR: No test data found in CSV!")
        return None, None
    
    texts = test_df['text'].tolist()
    labels = test_df['label'].tolist()
    
    print(f"Test set size: {len(texts)} samples")
    print(f"  Benign (label=0): {labels.count(0)}")
    print(f"  Phishing (label=1): {labels.count(1)}")
    
    return texts, labels


def predict_batch(texts, tokenizer, model, batch_size=32):
    """Predict on batch of texts and return predictions + probabilities"""
    all_preds = []
    all_probs = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of phishing class
    
    return all_preds, all_probs


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Phishing'],
                yticklabels=['Benign', 'Phishing'])
    plt.title('Confusion Matrix - BERT Text Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(labels, probs, save_path):
    """Plot and save ROC curve"""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - BERT Text Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(labels, probs, save_path):
    """Plot and save Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - BERT Text Model', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_distribution(labels, predictions, save_path):
    """Plot class distribution comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True distribution
    true_counts = [labels.count(0), labels.count(1)]
    ax1.bar(['Benign', 'Phishing'], true_counts, color=['#2ecc71', '#e74c3c'])
    ax1.set_title('True Class Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=10)
    for i, v in enumerate(true_counts):
        ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Predicted distribution
    pred_counts = [predictions.count(0), predictions.count(1)]
    ax2.bar(['Benign', 'Phishing'], pred_counts, color=['#3498db', '#f39c12'])
    ax2.set_title('Predicted Class Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=10)
    for i, v in enumerate(pred_counts):
        ax2.text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(labels, predictions, probs):
    """Generate detailed evaluation report"""
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    roc_auc = roc_auc_score(labels, probs)
    avg_precision = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, predictions)
    
    # Classification report
    class_names = ['Benign', 'Phishing']
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    
    # Per-class metrics
    tn, fp, fn, tp = cm.ravel()
    benign_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    benign_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    benign_f1 = 2 * (benign_precision * benign_recall) / (benign_precision + benign_recall) if (benign_precision + benign_recall) > 0 else 0
    benign_support = tn + fp
    
    phishing_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    phishing_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    phishing_f1 = 2 * (phishing_precision * phishing_recall) / (phishing_precision + phishing_recall) if (phishing_precision + phishing_recall) > 0 else 0
    phishing_support = fn + tp
    
    # Generate text report
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("TEXT MODEL EVALUATION REPORT")
    report_lines.append("="*60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("OVERALL METRICS:")
    report_lines.append("-"*60)
    report_lines.append(f"Accuracy:       {accuracy:.4f}")
    report_lines.append(f"Precision:      {precision:.4f}")
    report_lines.append(f"Recall:         {recall:.4f}")
    report_lines.append(f"F1-Score:       {f1:.4f}")
    report_lines.append(f"ROC-AUC:        {roc_auc:.4f}")
    report_lines.append(f"Avg Precision:  {avg_precision:.4f}")
    report_lines.append("")
    
    report_lines.append("CLASSIFICATION REPORT:")
    report_lines.append("-"*60)
    report_lines.append(report)
    report_lines.append("")
    
    report_lines.append("CONFUSION MATRIX:")
    report_lines.append("-"*60)
    report_lines.append("                Predicted")
    report_lines.append("              Benign  Phishing")
    report_lines.append(f"True Benign     {tn:6d}    {fp:6d}")
    report_lines.append(f"     Phishing   {fn:6d}    {tp:6d}")
    report_lines.append("")
    
    report_lines.append("PER-CLASS DETAILED METRICS:")
    report_lines.append("-"*60)
    report_lines.append("")
    report_lines.append("Benign:")
    report_lines.append(f"  Precision: {benign_precision:.4f}")
    report_lines.append(f"  Recall:    {benign_recall:.4f}")
    report_lines.append(f"  F1-Score:  {benign_f1:.4f}")
    report_lines.append(f"  Support:   {benign_support}")
    report_lines.append("")
    report_lines.append("Phishing:")
    report_lines.append(f"  Precision: {phishing_precision:.4f}")
    report_lines.append(f"  Recall:    {phishing_recall:.4f}")
    report_lines.append(f"  F1-Score:  {phishing_f1:.4f}")
    report_lines.append(f"  Support:   {phishing_support}")
    report_lines.append("")
    report_lines.append("="*60)
    
    return "\n".join(report_lines), cm, accuracy, precision, recall, f1, roc_auc, avg_precision


def evaluate():
    """Main evaluation function"""
    # Load model
    tokenizer, model = load_model()
    
    # Load test data
    texts, labels = load_test_data()
    if texts is None:
        return
    
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    # Get predictions and probabilities
    predictions, probs = predict_batch(texts, tokenizer, model)
    
    # Generate report
    report_text, cm, accuracy, precision, recall, f1, roc_auc, avg_precision = generate_report(labels, predictions, probs)
    
    # Print to console
    print("\n" + report_text)
    
    # Save report to file
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n✓ Report saved to {report_path}")
    
    # Generate and save plots
    print("\nGenerating visualizations...")
    
    # Confusion Matrix
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path)
    print(f"✓ Confusion matrix saved to {cm_path}")
    
    # ROC Curve
    roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
    plot_roc_curve(labels, probs, roc_path)
    print(f"✓ ROC curve saved to {roc_path}")
    
    # Precision-Recall Curve
    pr_path = os.path.join(OUTPUT_DIR, 'precision_recall_curve.png')
    plot_precision_recall_curve(labels, probs, pr_path)
    print(f"✓ Precision-Recall curve saved to {pr_path}")
    
    # Class Distribution
    dist_path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
    plot_class_distribution(labels, predictions, dist_path)
    print(f"✓ Class distribution saved to {dist_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    evaluate()