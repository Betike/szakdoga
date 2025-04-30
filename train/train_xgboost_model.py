import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split   

os.makedirs('models', exist_ok=True)
os.makedirs('utils', exist_ok=True)

np.random.seed(42)

train_data = pd.read_csv("data/train_test/train_data_chronological.csv")
test_data = pd.read_csv("data/train_test/test_data_chronological.csv")

print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

feature_cols = [col for col in train_data.columns 
                if ((col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_')) 
                and not col.endswith('_D') and not col.endswith('TotalD'))
            ]

for col in feature_cols.copy():
    if train_data[col].isna().sum() > 0 or test_data[col].isna().sum() > 0:
        feature_cols.remove(col)

X_train = train_data[feature_cols]
X_test = test_data[feature_cols]

label_encoder = LabelEncoder()
label_encoder.fit(train_data['Result'])
y_train = label_encoder.transform(train_data['Result'])
y_test = label_encoder.transform(test_data['Result'])
X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False)


label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

with open('utils/xgboost/xgboost_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
dval   = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': ['mlogloss', 'merror'],
    'eta': 0.05,
    'max_depth': 3,
    'min_child_weight': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'gamma': 0.1,
    'alpha': 0.5,
    'lambda': 1.0,
    'seed': 42
}

watchlist = [(dtrain, 'train'), (dval, 'val')]

print("\nTraining XGBoost model...")
num_rounds = 1000
early_stopping_rounds = 20

evals_result = {}

xgb_model = xgb.train(
    params, 
    dtrain, 
    num_rounds, 
    evals=watchlist,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=10,
    evals_result=evals_result
)


y_pred_prob = xgb_model.predict(dtest)
y_pred = np.argmax(y_pred_prob, axis=1)

y_pred_labels = [label_mapping[p] for p in y_pred]
y_true_labels = [label_mapping[t] for t in y_test]

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)

model_path = f"models/xgboost_prediction_model.json"
xgb_model.save_model(model_path)

feature_importance = xgb_model.get_score(importance_type='weight')
importance_df = pd.DataFrame({
    'Feature': list(feature_importance.keys()),
    'Importance': list(feature_importance.values())
}).sort_values('Importance', ascending=False)

importance_df.to_csv('utils/xgboost/xgboost_feature_importance.csv', index=False)

train_metrics = {
    'iteration': [],
    'train_mlogloss': [],
    'train_merror': [],
    'val_mlogloss': [],
    'val_merror': []
}

for i in range(len(evals_result['train']['mlogloss'])):
    train_metrics['iteration'].append(i)
    train_metrics['train_mlogloss'].append(evals_result['train']['mlogloss'][i])
    train_metrics['train_merror'].append(evals_result['train']['merror'][i])
    train_metrics['val_mlogloss'].append(evals_result['val']['mlogloss'][i])
    train_metrics['val_merror'].append(evals_result['val']['merror'][i])

metrics_df = pd.DataFrame(train_metrics)

metrics_df.to_csv('utils/xgboost/xgboost_training_metrics.csv', index=False)

plt.figure(figsize=(12, 8))
xgb.plot_importance(xgb_model, max_num_features=15, importance_type='weight', title='Feature Importance')
plt.tight_layout()
plt.savefig('visualisations/xgboost/xgboost_feature_importance.png')

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XGBoost Confusion Matrix')
plt.tight_layout()
plt.savefig('visualisations/xgboost/xgboost_confusion_matrix.png')

print("\nTraining complete.")