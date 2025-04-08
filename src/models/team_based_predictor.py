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

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_team_data import load_team_data, prepare_data_for_training

class TeamBasedPredictor(nn.Module):
    def __init__(self, input_size):
        super(TeamBasedPredictor, self).__init__()
        
        self.team_attr_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        self.form_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16)
        )
        
        self.combined_encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        home_attr = x[:, :6]
        home_form = x[:, 6:16]
        
        away_attr = x[:, 16:22]
        away_form = x[:, 22:32]
        
        home_attr_encoded = self.team_attr_encoder(home_attr)
        home_form_encoded = self.form_encoder(home_form)
        away_attr_encoded = self.team_attr_encoder(away_attr)
        away_form_encoded = self.form_encoder(away_form)
        
        combined = torch.cat([
            home_attr_encoded, home_form_encoded,
            away_attr_encoded, away_form_encoded
        ], dim=1)
        
        return self.combined_encoder(combined)
    
    def save_model(self, path):
        """Save the model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.team_attr_encoder[0].in_features * 4
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

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, save_path='model.pth', patience=10):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    classes, counts = np.unique(y_train, return_counts=True)
    weight_dict = {c: len(y_train) / (len(classes) * count) for c, count in zip(classes, counts)}
    class_weights = torch.FloatTensor([weight_dict[i] for i in range(len(classes))])
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        indices = torch.randperm(len(X_train))
        X_train_shuffled = X_train_tensor[indices]
        y_train_shuffled = y_train_tensor[indices]
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            
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
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Early stopping check
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            model.save_model(save_path)  # Save best model
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss.item():.4f}, '
                  f'Test Acc: {test_acc:.4f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Plot training progress
    plot_training_progress(train_losses, test_losses, train_accuracies, test_accuracies)
    
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
    plt.savefig('team_based_training_progress.png')
    plt.close()

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    print("\nModel Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, predicted.numpy(), 
                              target_names=['Away Win', 'Draw', 'Home Win']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, predicted.numpy())

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Draw', 'Home Win'],
                yticklabels=['Away Win', 'Draw', 'Home Win'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('team_based_confusion_matrix.png')
    plt.close()

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    print("Loading data...")
    X, y, feature_columns = load_team_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(X, y)
    
    # Initialize and train model
    print("\nTraining team-based model...")
    model = TeamBasedPredictor(input_size=len(feature_columns))
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, X_train, y_train, X_test, y_test, 
        save_path='models/team_based_predictor.pth',
        patience=10
    )
    
    
    evaluate_model(model, X_test, y_test)
    
   
    print("\nLoading saved model...")
    loaded_model = TeamBasedPredictor.load_model('models/team_based_predictor.pth')
    evaluate_model(loaded_model, X_test, y_test)

if __name__ == "__main__":
    main() 