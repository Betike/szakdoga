import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import os
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))
from data.prepare_match_data import load_match_data, prepare_data_for_training

class MatchPredictor(nn.Module):
    def __init__(self, input_size):
        super(MatchPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        return self.model(x)

    def save_model(self, path):
        """Save the model state and optimizer state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.model[0].in_features
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

def calculate_class_weights(y):
    """Calculate class weights for imbalanced data."""
    class_counts = Counter(y)
    total = len(y)
    weights = {cls: total / (len(class_counts) * count) 
              for cls, count in class_counts.items()}
    return torch.FloatTensor([weights[i] for i in range(len(class_counts))])

def plot_training_progress(train_losses, test_losses, train_accuracies, test_accuracies):
    """Plot training and test metrics."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.close()

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, save_path='model.pth', patience=10):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    class_weights = calculate_class_weights(y_train)
    
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
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
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
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            model.save_model(save_path)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Test Loss: {test_loss.item():.4f}, '
                  f'Test Acc: {test_acc:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    plot_training_progress(train_losses, test_losses, train_accuracies, test_accuracies)
    
    return train_losses, test_losses, train_accuracies, test_accuracies

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
    print("\nModel Evaluation:")
    print("\nClassification Report:")
    print(classification_report(y_test, predicted.numpy(), 
                              target_names=['Away Win', 'Draw', 'Home Win']))

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    df, feature_columns = load_match_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(df, feature_columns)
    
    model = MatchPredictor(input_size=len(feature_columns))
    print("\nTraining model...")
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, X_train, y_train, X_test, y_test, 
        save_path='models/match_predictor.pth',
        patience=10
    )
    
    evaluate_model(model, X_test, y_test)
    
    print("\nLoading saved model...")
    loaded_model = MatchPredictor.load_model('models/match_predictor.pth')
    evaluate_model(loaded_model, X_test, y_test) 