import pandas as pd
import numpy as np
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.model_selection import train_test_split  

# ---------------------------------------------------------------------
# 0.  Set‑up
# ---------------------------------------------------------------------
os.makedirs('models', exist_ok=True)
os.makedirs('utils', exist_ok=True)
np.random.seed(42)

# ---------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------
print("Loading prepared datasets...")
train_data = pd.read_csv("data/train_test/train_data_chronological.csv")
test_data  = pd.read_csv("data/train_test/test_data_chronological.csv")

print(f"Training data shape: {train_data.shape}")
print(f"Testing  data shape: {test_data.shape}")

# ---------------------------------------------------------------------
# 2.  Feature engineering
# ---------------------------------------------------------------------
print("\nPreparing features and target variables...")
feature_cols = [
    col for col in train_data.columns
    if (col.startswith('Home_') or col.startswith('Away_') or col.startswith('Diff_'))
]

for col in feature_cols.copy():            # drop columns that still have NaN
    if train_data[col].isna().any() or test_data[col].isna().any():
        print(f"Removing column with NaN values: {col}")
        feature_cols.remove(col)

print(f"Number of features used: {len(feature_cols)}")
print("Sample features:", feature_cols[:5])

X_train = train_data[feature_cols]
X_test  = test_data[feature_cols]

# ---------------------------------------------------------------------
# 3.  Encode labels
# ---------------------------------------------------------------------
label_encoder = LabelEncoder()
label_encoder.fit(train_data['Result'])

y_train = label_encoder.transform(train_data['Result'])
y_test  = label_encoder.transform(test_data['Result'])

X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, shuffle=False)

label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print(f"\nLabel encoding: {label_mapping}")

with open('utils/random_forest/random_forest_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('utils/random_forest/random_forest_features.json', 'w') as f:
    json.dump({'feature_names': feature_cols}, f)

# ---------------------------------------------------------------------
# 4.  Random‑forest model
# ---------------------------------------------------------------------
tune_hyperparameters = False   # switch to True if you want random search

if tune_hyperparameters:
    print("\nPerforming hyperparameter tuning for Random Forest...")
    param_grid = {
        'n_estimators':      [100, 150, 200, 300],
        'max_features':      [0.2, 0.3, 'sqrt'],
        'max_depth':         [6, 8, 10, 12, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf':  [2, 5, 8],
        'bootstrap':         [True],
        'class_weight':      ['balanced']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,                # 3-fold cross-validation
        verbose=1,
        n_jobs=-1,
        scoring='accuracy'
    )
    grid_search.fit(X_tr, y_tr)
    print(f"Best parameters: {grid_search.best_params_}")
    rf_model = grid_search.best_estimator_

else:
    print("\nTraining Random Forest model with revised parameters...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=5,
        max_features=0.3,
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_tr, y_tr)

# ---------------------------------------------------------------------
# 5.  Evaluation
# ---------------------------------------------------------------------
train_accuracy = rf_model.score(X_tr, y_tr)
print(f"\nTraining accuracy: {train_accuracy:.4f}")

val_accuracy = rf_model.score(X_val, y_val)
print(f"\nValidation accuracy (2017-23 hold-out): {val_accuracy:.4f}")

y_pred = rf_model.predict(X_test)

y_pred_labels = label_encoder.inverse_transform(y_pred)
y_true_labels = label_encoder.inverse_transform(y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
print("\nConfusion Matrix:")
print(conf_matrix)

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# ---------------------------------------------------------------------
# 6.  Persist artefacts
# ---------------------------------------------------------------------
model_path = f"models/random_forest_prediction_model.pkl"

with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)

print(f"\nModel saved to {model_path} and models/random_forest_prediction_model.pkl")

feature_importance.to_csv('utils/random_forest/random_forest_feature_importance.csv', index=False)

# ---------------------------------------------------------------------
# 7.  Visualisations
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig('visualisations/random_forest/random_forest_confusion_matrix.png')

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
plt.title('Random Forest - Top 20 Feature Importance')
plt.tight_layout()
plt.savefig('visualisations/random_forest/random_forest_feature_importance.png')

print("\nTraining complete! You can now use the Random Forest model for predictions.")
