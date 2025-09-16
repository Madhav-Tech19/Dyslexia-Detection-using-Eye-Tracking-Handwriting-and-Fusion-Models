import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Check GPU memory and optimize settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Optimize for GTX 1050 3GB
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Clear cache at start
    torch.cuda.empty_cache()

class MultiTaskDataset(Dataset):
    def __init__(self, task_data, labels, transform=None):
        self.task_data = task_data
        self.labels = labels
        self.transform = transform
        self.common_subjects = self._find_common_subjects()
        print(f"Found {len(self.common_subjects)} subjects with data in all tasks")
        
    def _find_common_subjects(self):
        task_subjects = {}
        for task, data in self.task_data.items():
            subjects = []
            for img_path in data['image_paths']:
                filename = os.path.basename(img_path)
                if filename.startswith('Subject_'):
                    subject_id = filename.split('_')[1]
                    subjects.append(subject_id)
            task_subjects[task] = set(subjects)
        
        if task_subjects:
            common = set.intersection(*task_subjects.values())
            return list(common)
        return []
    
    def __len__(self):
        return len(self.common_subjects)
    
    def __getitem__(self, idx):
        subject_id = self.common_subjects[idx]
        images = {}
        
        for task, data in self.task_data.items():
            task_img = None
            for img_path in data['image_paths']:
                filename = os.path.basename(img_path)
                if filename.startswith('Subject_'):
                    path_subject_id = filename.split('_')[1]
                    if path_subject_id == subject_id:
                        try:
                            task_img = Image.open(img_path).convert('RGB')
                            break
                        except:
                            continue
            
            if task_img is None:
                task_img = Image.new('RGB', (224, 224), color='black')
            
            if self.transform:
                task_img = self.transform(task_img)
            
            images[task] = task_img
        
        label = self.labels.get(subject_id, 0)
        label = torch.tensor(label, dtype=torch.long)
        return images, label

class LightweightDyslexiaClassifier(nn.Module):
    """Lightweight model optimized for 3GB GPU"""
    def __init__(self, num_classes=2, pretrained=True):
        super(LightweightDyslexiaClassifier, self).__init__()
        
        from torchvision import models
        # Use MobileNetV2 for better memory efficiency
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.feature_extractor = mobilenet.features
        
        # Reduced feature dimension for memory efficiency
        feature_dim = 1280  # MobileNetV2 output
        
        # Simplified task processors
        self.task_processors = nn.ModuleDict({
            'T1_Syllables': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128)
            ),
            'T4_Meaningful_Text': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128)
            ),
            'T5_Pseudo_Text': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128)
            )
        })
        
        # Simplified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.Tanh(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # Simplified fusion network
        self.fusion = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        self.feature_output = nn.Linear(64, 32)
        
    def forward(self, task_images, return_features=False):
        batch_size = next(iter(task_images.values())).size(0)
        task_features = []
        
        for task in ['T1_Syllables', 'T4_Meaningful_Text', 'T5_Pseudo_Text']:
            if task in task_images:
                img_features = self.feature_extractor(task_images[task])
                task_feat = self.task_processors[task](img_features)
                task_features.append(task_feat)
            else:
                zero_feat = torch.zeros(batch_size, 128).to(next(self.parameters()).device)
                task_features.append(zero_feat)
        
        combined_features = torch.cat(task_features, dim=1)
        
        attention_weights = self.attention(combined_features)
        
        weighted_features = []
        for i in range(3):
            weighted_feat = task_features[i] * attention_weights[:, i:i+1]
            weighted_features.append(weighted_feat)
        
        final_combined = torch.cat(weighted_features, dim=1)
        fused_features = self.fusion(final_combined)
        
        logits = self.classifier(fused_features)
        
        if return_features:
            feature_vector = self.feature_output(fused_features)
            return logits, feature_vector
        
        return logits

def load_labels(data_dir):
    labels_path = os.path.join(data_dir, 'dyslexia_class_label.csv')
    print(f"Loading labels from: {labels_path}")
    
    if not os.path.exists(labels_path):
        print(f"Labels file not found!")
        return {}, {}, {}
    
    labels_df = pd.read_csv(labels_path)
    print(f"Labels shape: {labels_df.shape}")
    print(f"Columns: {labels_df.columns.tolist()}")
    
    subject_col = labels_df.columns[0]
    label_col = labels_df.columns[1]
    
    labels_df[subject_col] = labels_df[subject_col].astype(str)
    unique_labels = labels_df[label_col].unique()
    
    label_mapping = {}
    for i, label in enumerate(sorted(unique_labels)):
        label_mapping[label] = i
    
    labels_df['numeric'] = labels_df[label_col].map(label_mapping)
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    print(f"Label mapping: {label_mapping}")
    
    return dict(zip(labels_df[subject_col], labels_df['numeric'])), label_mapping, reverse_mapping

def prepare_task_data(data_dir):
    task_data = {}
    task_mapping = {
        'T1_Syllables': {'directories': ['Syllables'], 'suffixes': ['_Syllables']},
        'T4_Meaningful_Text': {'directories': ['MeaningfulText'], 'suffixes': ['_MeaningfulText']},
        'T5_Pseudo_Text': {'directories': ['PseudoText'], 'suffixes': ['_PseudoText']}
    }
    
    fixation_images_dir = os.path.join(data_dir, 'fixation_images')
    
    for task in ['T1_Syllables', 'T4_Meaningful_Text', 'T5_Pseudo_Text']:
        mapping = task_mapping[task]
        image_paths = []
        
        for dir_name in mapping['directories']:
            task_images_dir = os.path.join(fixation_images_dir, dir_name)
            if os.path.exists(task_images_dir):
                available_images = os.listdir(task_images_dir)
                for img_file in available_images:
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(task_images_dir, img_file))
                break
        
        task_data[task] = {'image_paths': image_paths, 'labels': []}
        print(f"Task {task}: found {len(image_paths)} images")
    
    return task_data

def plot_training_history(train_losses, train_accuracies, val_accuracies, save_path='training_history.png'):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
    plt.title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1])
    
    plt.subplot(1, 3, 3)
    plt.plot(val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Training history plots saved to: {save_path}")

def generate_classification_report_and_confusion_matrix(model, val_loader, reverse_mapping, device):
    print("\n" + "="*60)
    print("GENERATING CLASSIFICATION REPORT AND CONFUSION MATRIX")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for task_images, targets in val_loader:
            for task in task_images:
                task_images[task] = task_images[task].to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(task_images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    class_names = [str(reverse_mapping[i]) for i in sorted(reverse_mapping.keys())]
    
    print("\n📊 CLASSIFICATION REPORT:")
    print("="*50)
    report = classification_report(all_targets, all_predictions, target_names=class_names, digits=4)
    print(report)
    
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    cm = confusion_matrix(all_targets, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')
    
    print(f"\n📈 ADDITIONAL METRICS:")
    print("="*30)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    
    results_summary = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'all_predictions': all_predictions.tolist(),
        'all_targets': all_targets.tolist()
    }
    
    with open('validation_metrics.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    results_df = pd.DataFrame({
        'true_label': all_targets,
        'predicted_label': all_predictions,
        'true_class': [class_names[i] for i in all_targets],
        'predicted_class': [class_names[i] for i in all_predictions],
        'correct_prediction': all_targets == all_predictions
    })
    results_df.to_csv('validation_results.csv', index=False)
    
    print(f"✅ Classification report saved to: classification_report.txt")
    print(f"✅ Confusion matrix saved to: confusion_matrix.png")
    print(f"✅ Validation metrics saved to: validation_metrics.json")
    print(f"✅ Validation results CSV saved to: validation_results.csv")
    
    return accuracy, precision, recall, f1

def train_model(data_dir, epochs=100, batch_size=4):  # Reduced batch size for 3GB GPU
    print("="*60)
    print("STARTING DYSLEXIA CLASSIFICATION TRAINING")
    print("="*60)
    
    labels_dict, label_mapping, reverse_mapping = load_labels(data_dir)
    task_data = prepare_task_data(data_dir)
    
    # Optimized transforms for memory efficiency
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Direct resize instead of crop
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = MultiTaskDataset(task_data, labels_dict, train_transforms)
    
    if len(dataset) == 0:
        print("No data found!")
        return None, None, None
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transforms
    
    # Optimized DataLoader settings for Windows
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=False  # Disable pin_memory for 3GB GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        pin_memory=False  # Disable pin_memory for 3GB GPU
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Use lightweight model
    model = LightweightDyslexiaClassifier(num_classes=2).to(device)
    
    # Remove mixed precision for GTX 1050 compatibility
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True
    )
    
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience = 30
    patience_counter = 0
    save_path = 'lightweight_dyslexia_model.pth'
    
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (task_images, targets) in enumerate(train_loader):
            # Clear GPU cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            for task in task_images:
                task_images[task] = task_images[task].to(device, non_blocking=False)
            targets = targets.to(device, non_blocking=False)
            
            optimizer.zero_grad()
            
            outputs = model(task_images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            if batch_idx % 5 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: Loss {loss.item():.4f}')
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for task_images, targets in val_loader:
                for task in task_images:
                    task_images[task] = task_images[task].to(device, non_blocking=False)
                targets = targets.to(device, non_blocking=False)
                
                outputs = model(task_images)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
            
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f'  GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_mapping': label_mapping,
                'reverse_label_mapping': reverse_mapping,
                'val_accuracy': val_acc,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, save_path)
            print(f'  ✓ New best model saved! Val Acc: {val_acc:.4f}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            break
        
        scheduler.step(val_acc)
        
        # Clear cache every few epochs
        if epoch % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f'\n🎉 Training completed!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    plot_training_history(train_losses, train_accuracies, val_accuracies)
    
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_accuracy, precision, recall, f1 = generate_classification_report_and_confusion_matrix(
        model, val_loader, reverse_mapping, device
    )
    
    return model, label_mapping, reverse_mapping

def generate_predictions(model, task_data, label_mapping, reverse_mapping, data_dir):
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS AND FEATURE VECTORS")
    print("="*60)
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    model.eval()
    results = []
    
    subject_data = {}
    for task in task_data:
        for img_path in task_data[task]['image_paths']:
            filename = os.path.basename(img_path)
            if filename.startswith('Subject_'):
                subject_id = filename.split('_')[1]
                if subject_id not in subject_data:
                    subject_data[subject_id] = {}
                subject_data[subject_id][task] = img_path
    
    complete_subjects = {sid: data for sid, data in subject_data.items() if len(data) == 3}
    print(f"Processing {len(complete_subjects)} subjects with complete data...")
    
    with torch.no_grad():
        for subject_id, task_images_paths in complete_subjects.items():
            task_images = {}
            
            for task, img_path in task_images_paths.items():
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = val_transforms(image).unsqueeze(0)
                    task_images[task] = image.to(device, non_blocking=False)
                except:
                    blank_image = torch.zeros(1, 3, 224, 224).to(device, non_blocking=False)
                    task_images[task] = blank_image
            
            logits, feature_vector = model(task_images, return_features=True)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).cpu().numpy()[0]
            confidence = torch.max(probabilities, dim=1)[0].cpu().numpy()[0]
            
            class_name = reverse_mapping.get(predicted_class, f"Unknown_{predicted_class}")
            
            result = {
                'subject_id': subject_id,
                'predicted_class': class_name,
                'predicted_numeric': predicted_class,
                'confidence': confidence,
                'prob_class_0': probabilities[0][0].cpu().numpy(),
                'prob_class_1': probabilities[0][1].cpu().numpy()
            }
            
            feature_vec = feature_vector.cpu().numpy()[0]
            for i, feat_val in enumerate(feature_vec):
                result[f'feature_{i:02d}'] = feat_val
            
            results.append(result)
    
    results_df = pd.DataFrame(results)
    output_file = 'dyslexia_predictions_with_vectors.csv'
    results_df.to_csv(output_file, index=False)
    
    model_info = {
        'model_path': 'lightweight_dyslexia_model.pth',
        'best_accuracy': float(results_df['confidence'].mean()),
        'label_mapping': {str(k): int(v) for k, v in label_mapping.items()},
        'feature_vector_size': 32,  # Updated for lightweight model
        'total_subjects': int(len(results_df))
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Predictions saved to: {output_file}")
    print(f"✅ Model info saved to: model_info.json")
    print(f"📊 Results shape: {results_df.shape}")
    print(f"📈 Prediction distribution:")
    print(results_df['predicted_class'].value_counts())
    print(f"🎯 Average confidence: {results_df['confidence'].mean():.3f}")
    
    return results_df

if __name__ == '__main__':  # THIS IS CRUCIAL FOR WINDOWS MULTIPROCESSING
    print("🚀 Starting ETDD70 Dyslexia Classification Pipeline...")
    
    data_dir = "D:\\MSCProject\\ETDD70"
    
    model, label_mapping, reverse_mapping = train_model(data_dir, epochs=100, batch_size=4)
    
    if model is not None:
        task_data = prepare_task_data(data_dir)
        
        predictions_df = generate_predictions(model, task_data, label_mapping, reverse_mapping, data_dir)
        
        print("\n🎊 SUCCESS! Training and prediction completed!")
        print("📁 Files created:")
        print("   - lightweight_dyslexia_model.pth (trained model)")
        print("   - dyslexia_predictions_with_vectors.csv (predictions + 32D vectors)")
        print("   - validation_results.csv (validation results)")
        print("   - model_info.json (model metadata)")
        print("   - training_history.png (training plots)")
        print("   - confusion_matrix.png (confusion matrix)")
        print("   - classification_report.txt (detailed metrics)")
        print("   - validation_metrics.json (validation metrics)")
    
    else:
        print("❌ Training failed - no data found!")