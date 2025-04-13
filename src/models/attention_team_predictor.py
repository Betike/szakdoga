import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import sys
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_team_data import load_team_data, prepare_data_for_training

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim]))
        
    def forward(self, x):
        # x shape: [batch_size, feature_dim]
        batch_size = x.shape[0]
        
        # Create query, key, and value projections
        Q = self.query(x)  # [batch_size, feature_dim]
        K = self.key(x)    # [batch_size, feature_dim]
        V = self.value(x)  # [batch_size, feature_dim]
        
        # Calculate attention scores
        attention = torch.matmul(Q.unsqueeze(1), K.unsqueeze(2)) / self.scale
        attention = torch.softmax(attention.squeeze(-1), dim=1)
        
        # Apply attention weights to values
        weighted_values = attention.unsqueeze(2) * V.unsqueeze(1)
        output = weighted_values.sum(dim=1)
        
        return output, attention

class AttentionTeamPredictor(nn.Module):
    def __init__(self, input_size):
        super(AttentionTeamPredictor, self).__init__()
        
        # Feature dimensions
        self.home_attr_dim = 6  # Home team attributes
        self.away_attr_dim = 6  # Away team attributes
        self.home_form_dim = 10 # Home team form
        self.away_form_dim = 10 # Away team form
        
        # Feature transformation layers
        self.home_attr_transform = nn.Sequential(
            nn.Linear(self.home_attr_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.away_attr_transform = nn.Sequential(
            nn.Linear(self.away_attr_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.home_form_transform = nn.Sequential(
            nn.Linear(self.home_form_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.away_form_transform = nn.Sequential(
            nn.Linear(self.away_form_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Home team self-attention
        self.home_attention = SelfAttention(64)
        
        # Away team self-attention
        self.away_attention = SelfAttention(64)
        
        # Cross-team attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # Final prediction layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Store attention weights for visualization
        self.home_attention_weights = None
        self.away_attention_weights = None
        self.cross_attention_weights = None
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Split input into components
        home_attr = x[:, :6]
        home_form = x[:, 6:16]
        away_attr = x[:, 16:22]
        away_form = x[:, 22:32]
        
        # Transform features
        home_attr_features = self.home_attr_transform(home_attr)
        away_attr_features = self.away_attr_transform(away_attr)
        home_form_features = self.home_form_transform(home_form)
        away_form_features = self.away_form_transform(away_form)
        
        # Apply home team attention (between attributes and form)
        home_features = torch.cat([
            home_attr_features.unsqueeze(1),
            home_form_features.unsqueeze(1)
        ], dim=1)  # [batch_size, 2, 64]
        
        # Apply away team attention
        away_features = torch.cat([
            away_attr_features.unsqueeze(1),
            away_form_features.unsqueeze(1)
        ], dim=1)  # [batch_size, 2, 64]
        
        # Create team representations by combining attributes and form
        home_combined = torch.cat([home_attr_features, home_form_features], dim=1)
        away_combined = torch.cat([away_attr_features, away_form_features], dim=1)
        
        # Cross-team attention
        team_features = torch.cat([
            home_combined.unsqueeze(1),
            away_combined.unsqueeze(1)
        ], dim=1)  # [batch_size, 2, 128]
        
        # Apply multi-head attention across teams
        cross_team_attn, cross_attn_weights = self.cross_attention(
            team_features, team_features, team_features
        )
        self.cross_attention_weights = cross_attn_weights
        
        # Flatten and concatenate for final prediction
        cross_team_flat = cross_team_attn.reshape(cross_team_attn.size(0), -1)
        
        # Concatenate with original features for residual connection
        combined_features = torch.cat([
            cross_team_flat, 
            home_combined,
            away_combined
        ], dim=1)
        
        # Final prediction
        output = self.classifier(combined_features)
        
        return output
    
    def get_attention_weights(self):
        """Return the attention weights from the last forward pass."""
        return {
            'cross_team': self.cross_attention_weights
        }
    
    def save_model(self, path):
        """Save the model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': 32  # Total input features (6+10+6+10)
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path):
        """Load a saved model."""
        checkpoint = torch.load(path)
        model = cls(input_size=checkpoint['input_size'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return model
    
    def summary(self):
        """Print a summary of the model architecture."""
        print("\nAttention Team Predictor Architecture:")
        print("=" * 50)
        print("Feature Transformation Layers:")
        print(f"  Home Team Attributes: {self.home_attr_transform}")
        print(f"  Home Team Form: {self.home_form_transform}")
        print(f"  Away Team Attributes: {self.away_attr_transform}")
        print(f"  Away Team Form: {self.away_form_transform}")
        
        print("\nAttention Mechanisms:")
        print(f"  Cross-Team Attention: MultiheadAttention(embed_dim=128, num_heads=4)")
        
        print("\nFeature Dimensions:")
        print("  Home Team Features: 128 (64 attr + 64 form)")
        print("  Away Team Features: 128 (64 attr + 64 form)")
        print("  Cross-Attention Output: 256 (2x128)")
        print("  Combined Features: 512 (256 cross-attention + 128 home + 128 away)")
        
        print("\nClassifier:")
        for i, layer in enumerate(self.classifier):
            print(f"  Layer {i}: {layer}")
        print("=" * 50)

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, save_path='model.pth', patience=15):
    """Train the model with visualized progress."""
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Display dataset information
    print(f"\nTraining dataset size: {len(X_train)} samples")
    print(f"Test dataset size: {len(X_test)} samples")
    
    # Calculate and display class distribution
    class_counts = np.bincount(y_train)
    print("\nClass distribution in training data:")
    for cls in range(len(class_counts)):
        class_name = ['Away Win', 'Draw', 'Home Win'][cls]
        print(f"  {class_name}: {class_counts[cls]} samples ({class_counts[cls]/len(y_train)*100:.2f}%)")
    
    # Calculate class weights for imbalanced data
    weight_dict = {c: len(y_train) / (len(class_counts) * count) for c, count in enumerate(class_counts)}
    class_weights = torch.FloatTensor([weight_dict[i] for i in range(len(class_counts))])
    
    print("\nClass weights for training:")
    for i, weight in enumerate(['Away Win', 'Draw', 'Home Win']):
        print(f"  {weight}: {class_weights[i]:.4f}")
    
    # Use Label Smoothing for better generalization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW optimizer with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, amsgrad=True)
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    
    print("\nTraining configuration:")
    print(f"  Optimizer: AdamW (lr=0.001, weight_decay=1e-4)")
    print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    print(f"  Criterion: CrossEntropyLoss with label smoothing (0.1)")
    print(f"  Batch size: {batch_size}")
    print(f"  Total epochs: {epochs} (with early stopping patience={patience})")
    
    print("\nStarting training...")
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training Progress")
    
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Shuffle training data for each epoch
        shuffle_idx = torch.randperm(len(X_train_tensor))
        X_train_shuffled = X_train_tensor[shuffle_idx]
        y_train_shuffled = y_train_tensor[shuffle_idx]
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_loss = epoch_loss / (len(X_train) / batch_size)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            _, predicted = torch.max(test_outputs.data, 1)
            test_acc = accuracy_score(y_test, predicted.numpy())
            test_losses.append(test_loss.item())
            test_accuracies.append(test_acc)
        
        scheduler.step()
        
        # Update best model if test loss improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            model.save_model(save_path)
            improvement = "✓ (saved)"
        else:
            patience_counter += 1
            improvement = "✗"
        
        # Update progress bar with current metrics
        epoch_pbar.set_postfix({
            'Train Loss': f"{train_loss:.4f}",
            'Train Acc': f"{train_acc:.4f}",
            'Test Loss': f"{test_loss.item():.4f}",
            'Test Acc': f"{test_acc:.4f}",
            'Improved': improvement
        })
        
        # Print detailed metrics for every epoch
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss.item():.4f}, '
              f'Test Acc: {test_acc:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}, '
              f'Improved: {improvement}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Plot training progress
    plot_training_progress(train_losses, test_losses, train_accuracies, test_accuracies)
    
    # Final evaluation and confusion matrix
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor)
        _, final_predicted = torch.max(final_outputs.data, 1)
    
    plot_confusion_matrix(y_test, final_predicted.numpy())
    
    return train_losses, test_losses, train_accuracies, test_accuracies

def plot_training_progress(train_losses, test_losses, train_accuracies, test_accuracies):
    """Plot training and test metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    vis_dir = Path(__file__).parent.parent.parent / 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(vis_dir / 'attention_team_training_progress.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Draw', 'Home Win'],
                yticklabels=['Away Win', 'Draw', 'Home Win'])
    plt.title('Confusion Matrix - Attention Team Predictor')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    vis_dir = Path(__file__).parent.parent.parent / 'visualizations'
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(vis_dir / 'attention_team_confusion_matrix.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    print("\nModel Evaluation:")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, predicted.numpy(), 
                              target_names=['Away Win', 'Draw', 'Home Win']))
    
    # Calculate accuracy by class
    class_accuracies = {}
    for cls in range(3):
        class_mask = y_test == cls
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(y_test[class_mask], predicted.numpy()[class_mask])
            class_name = ['Away Win', 'Draw', 'Home Win'][cls]
            class_accuracies[class_name] = class_acc
    
    print("\nAccuracy by class:")
    for class_name, acc in class_accuracies.items():
        print(f"  {class_name}: {acc:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, predicted.numpy())
    print("=" * 50)

def main():
    # Create models directory if it doesn't exist
    model_dir = Path(__file__).parent.parent.parent / 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    print("\n========== Attention-Based Team Predictor ==========")
    print("Loading team data...")
    X, y, feature_columns = load_team_data()
    
    print(f"Loaded {len(X)} matches with {len(feature_columns)} features")
    print("\nPrepping data for training...")
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(X, y)
    
    print("\nInitializing Attention-Based model...")
    model = AttentionTeamPredictor(input_size=len(feature_columns))
    # Display model architecture
    model.summary()
    
    print("\nTraining attention-based model...")
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, X_train, y_train, X_test, y_test,
        save_path=model_dir / 'attention_team_predictor.pth',
        patience=15
    )
    
    print("\nEvaluating trained model...")
    evaluate_model(model, X_test, y_test)
    
    print("\nLoading saved model for verification...")
    loaded_model = AttentionTeamPredictor.load_model(model_dir / 'attention_team_predictor.pth')
    evaluate_model(loaded_model, X_test, y_test)
    print("\n===================================================")

if __name__ == "__main__":
    main() 