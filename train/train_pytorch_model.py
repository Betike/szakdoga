import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import copy
import random, os
from sklearn.model_selection import train_test_split

random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

os.makedirs('models', exist_ok=True)
os.makedirs('utils', exist_ok=True)

class FootballMatchDataset(Dataset):
    def __init__(self, features, targets, scaler=None, fit_scaler=False):
        if fit_scaler:
            self.scaler = StandardScaler()
            self.features = torch.FloatTensor(self.scaler.fit_transform(features))
        elif scaler is not None:
            self.scaler = scaler
            self.features = torch.FloatTensor(self.scaler.transform(features))
        else:
            self.scaler = None
            self.features = torch.FloatTensor(features.values)
            
        label_map = {'A': 0, 'D': 1, 'H': 2}
        self.targets = torch.LongTensor([label_map[t] for t in targets.values])
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class MatchPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.4):
        super(MatchPredictionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 3)  # 3 classes: home win, away win, draw
        )
    
    def forward(self, x):
        return self.model(x)

train_data = pd.read_csv("data/train_test/train_data_chronological.csv")
test_data = pd.read_csv("data/train_test/test_data_chronological.csv")

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

feature_cols = [col for col in train_data.columns 
                if (col.startswith('Diff_') or col.startswith('Home_') or col.startswith('Away_'))]

for col in feature_cols.copy():
    if train_data[col].isna().sum() > 0 or test_data[col].isna().sum() > 0:
        feature_cols.remove(col)

print(f"Features: {len(feature_cols)}")

X_train = train_data[feature_cols]
y_train = train_data['Result']

X_test = test_data[feature_cols]
y_test = test_data['Result']

X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False)

scaler = StandardScaler().fit(X_tr)

train_dataset = FootballMatchDataset(X_tr, y_tr, scaler=scaler)
val_dataset   = FootballMatchDataset(X_val, y_val, scaler=scaler)

test_dataset = FootballMatchDataset(X_test, y_test, scaler=scaler)

scaling_info = {
    'feature_names': feature_cols,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist()
}

with open('utils/neural_network/feature_scaling_info.json', 'w') as f:
    json.dump(scaling_info, f)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size) 

input_size = len(feature_cols)
model = MatchPredictionNN(input_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # L2 regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

num_epochs = 100
print(f"\nTraining for {num_epochs} epochs")

train_losses = []
train_accuracies = []
val_accuracies = []
best_accuracy = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_features, batch_targets in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_targets.size(0)
        train_correct += (predicted == batch_targets).sum().item()
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_features.size(0)
    

    epoch_loss = running_loss / len(train_dataset)
    train_accuracy = train_correct / train_total
    
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_targets.cpu().numpy())
    
    val_accuracy = accuracy_score(all_targets, all_preds)
    val_accuracies.append(val_accuracy)
    
    scheduler.step(val_accuracy)
    current_lr = scheduler.get_last_lr()[0]
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:3d}\t{epoch_loss:.4f}\t{train_accuracy:.4f}\t{val_accuracy:.4f}\t{current_lr:.6f}")

print(f"\nTraining completed. Best validation accuracy: {best_accuracy:.4f}")

model.load_state_dict(best_model_state)

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_features, batch_targets in test_loader:
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(batch_targets.cpu().numpy())

label_map_inverse = {0: 'A', 1: 'D', 2: 'H'}
y_pred = [label_map_inverse[p] for p in all_preds]
y_true = [label_map_inverse[t] for t in all_targets]

accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

conf_matrix = confusion_matrix(y_true, y_pred)

model_path = f"models/pytorch_model.pth"
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")

metrics_df = pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'train_loss': train_losses,
    'train_accuracy': train_accuracies,
    'val_accuracy': val_accuracies
})
metrics_df.to_csv('utils/neural_network/training_metrics.csv', index=False)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 2)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 3)
final_train_acc = train_accuracies[-1]
final_val_acc = val_accuracies[-1]
plt.bar(['Training', 'Validation'], [final_train_acc, final_val_acc])
plt.ylim(0, 1)
plt.ylabel('Final Accuracy')
for i, v in enumerate([final_train_acc, final_val_acc]):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('visualisations/neural_network/training_history.png')

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Away', 'Draw', 'Home'],
            yticklabels=['Away', 'Draw', 'Home'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('visualisations/neural_network/confusion_matrix.png')

print("\nTraining complete.")