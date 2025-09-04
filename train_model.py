import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# 1) Load dataset
data = pd.read_csv("diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 2) Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    ))
]

# 4) Meta learner (stacking)
meta_model = LogisticRegression(max_iter=1000)

# 5) Build hybrid ensemble
model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5
)

# 6) Train
model.fit(X_train, y_train)

# 7) Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"✅ Hybrid Model trained. Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# 8) Save model
pickle.dump(model, open("model.pkl", "wb"))
print("💾 Hybrid model saved as model.pkl")
