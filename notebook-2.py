# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load processed data
X = pd.read_csv('processed_train.csv')
test = pd.read_csv('processed_test.csv')

# Prepare data for modeling
y = X['TARGET']
X = X.drop(columns=['TARGET', 'SK_ID_CURR'])

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
lr_auc = roc_auc_score(y_test, y_pred_lr)

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test, y_pred_gb)

# Model evaluation
print("Logistic Regression AUC:", lr_auc)
print("Gradient Boosting AUC:", gb_auc)

# Feature importance for Logistic Regression
lr_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs(lr.coef_[0])
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=lr_importance.head(10))
plt.title('Top 10 Features - Logistic Regression')
plt.savefig('lr_feature_importance.png')

# Feature importance for Gradient Boosting
gb_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=gb_importance.head(10))
plt.title('Top 10 Features - Gradient Boosting')
plt.savefig('gb_feature_importance.png')

# ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_gb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {gb_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curves.png')

# Save models
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(gb, 'gradient_boosting_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Make predictions on test set
test_ids = test['SK_ID_CURR']
test = test.drop(columns=['SK_ID_CURR'])
test_scaled = scaler.transform(test)

lr_preds = lr.predict_proba(test_scaled)[:, 1]
gb_preds = gb.predict_proba(test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'SK_ID_CURR': test_ids,
    'TARGET': gb_preds  # Using gradient boosting predictions
})
submission.to_csv('submission.csv', index=False)