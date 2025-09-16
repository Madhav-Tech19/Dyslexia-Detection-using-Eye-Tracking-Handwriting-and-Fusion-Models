import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device for Kaggle GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

class DyslexiaDataset(Dataset):
    """Custom dataset for dyslexia detection"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Class mapping - only normal and reversal
        self.class_map = {'normal': 0, 'reversal': 1}
        self.class_names = ['normal', 'reversal']
        
        # Load images and labels
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.class_map:
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_map[class_name])
        
        print(f"Loaded {len(self.images)} images from {data_dir}")
        print(f"Class distribution: {pd.Series(self.labels).value_counts().to_dict()}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0))), label
            return Image.new('RGB', (224, 224), (0, 0, 0)), label

class LeViTDyslexiaClassifier(nn.Module):
    """LeViT model for dyslexia detection"""
    
    def __init__(self, model_name='levit_256', num_classes=2, pretrained=True):
        super(LeViTDyslexiaClassifier, self).__init__()
        
        # Load pretrained LeViT model
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the number of features from the backbone
        # For LeViT, we need to get the features after removing the head
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def get_transforms():
    """Get data transforms for training and validation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(train_dir, test_dir, batch_size=16):
    """Create data loaders for training and testing"""
    
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = DyslexiaDataset(train_dir, transform=train_transform)
    test_dataset = DyslexiaDataset(test_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader

def train_model(model, train_loader, val_loader, num_epochs=2):
    """Train the LeViT model"""
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Best Val Acc: {best_val_acc:.2f}%')
        print('-' * 50)
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                 target_names=['normal', 'reversal'],
                                 output_dict=True)
    
    return accuracy, report, all_preds, all_labels, all_probs

def plot_training_history(history):
    """Plot training history"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['val_accs'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['normal', 'reversal'],
                yticklabels=['normal', 'reversal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def save_results(predictions, labels, probabilities, output_file='dyslexia_results.csv'):
    """Save results for future decision making"""
    
    results_df = pd.DataFrame({
        'true_label': ['normal' if x == 0 else 'reversal' for x in labels],
        'predicted_label': ['normal' if x == 0 else 'reversal' for x in predictions],
        'probability_normal': [prob[0] for prob in probabilities],
        'probability_reversal': [prob[1] for prob in probabilities],
        'confidence': [max(prob) for prob in probabilities]
    })
    
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results_df

def main():
    """Main function to run the dyslexia detection pipeline"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data directories
    train_dir = "D:\MSCProject\Processed\Train"
    test_dir = "D:\MSCProject\Processed\Test"
    
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    
    print("=" * 60)
    print("DYSLEXIA DETECTION WITH LeViT - NORMAL vs REVERSAL")
    print("=" * 60)
    
    # Create data loaders
    print("\n1. Loading data...")
    train_loader, test_loader = create_data_loaders(train_dir, test_dir, batch_size)
    
    # Create model
    print("\n2. Creating LeViT model...")
    model = LeViTDyslexiaClassifier(model_name='levit_256', num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n3. Training model...")
    model, history = train_model(model, train_loader, test_loader, num_epochs)
    
    # Plot training history
    print("\n4. Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    accuracy, report, predictions, labels, probabilities = evaluate_model(model, test_loader)
    
    # Print results
    print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 60)
    for class_name in ['normal', 'reversal']:
        metrics = report[class_name]
        print(f"{class_name:<15} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
              f"{metrics['f1-score']:<10.4f} {metrics['support']:<10.0f}")
    
    # Plot confusion matrix
    print("\n6. Plotting confusion matrix...")
    plot_confusion_matrix(labels, predictions)
    
    # Save results
    print("\n7. Saving results...")
    results_df = save_results(predictions, labels, probabilities)
    
    # Save model
    print("\n8. Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': 'levit_256',
        'num_classes': 2,
        'accuracy': accuracy,
        'history': history
    }, 'dyslexia_levit_model.pth')
    print("Model saved to dyslexia_levit_model.pth")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"Best Validation Accuracy: {history['best_val_acc']:.2f}%")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"Model saved: dyslexia_levit_model.pth")
    print(f"Results saved: dyslexia_results.csv")
    print("="*60)
    
    return model, results_df

# Function to load and use saved model for inference
def load_model_for_inference(model_path='dyslexia_levit_model.pth'):
    """Load saved model for inference"""
    
    checkpoint = torch.load(model_path, map_location=device)
    model = LeViTDyslexiaClassifier(
        model_name=checkpoint['model_name'],
        num_classes=checkpoint['num_classes'],
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def predict_single_image(model, image_path):
    """Predict dyslexia for a single image"""
    
    _, val_transform = get_transforms()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = val_transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        probability = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
    
    class_names = ['normal', 'reversal']
    result = {
        'predicted_class': class_names[predicted_class],
        'probability_normal': probability[0][0].item(),
        'probability_reversal': probability[0][1].item(),
        'confidence': probability[0][predicted_class].item()
    }
    
    return result

if __name__ == "__main__":
    # Run the complete pipeline
    model, results_df = main()
    
    # Example of using the trained model for inference
    print("\nExample inference:")
    print("To use the model for inference on a new image:")
    print("model = load_model_for_inference('dyslexia_levit_model.pth')")
    print("result = predict_single_image(model, 'path/to/your/image.jpg')")
    print("print(result)")