import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# LOAD DATASET
# ============================================================

print("="*50)
print("Loading Dataset...")
print("="*50)

df = pd.read_csv("dataset/Training.csv")

print("Original Shape :", df.shape)

# ============================================================
# CLEAN DATA
# ============================================================

# Remove last empty column if present

if "Unnamed: 133" in df.columns:
    df.drop("Unnamed: 133", axis=1, inplace=True)

print("Cleaned Shape :", df.shape)

# ============================================================
# FEATURES & TARGET
# ============================================================

X = df.drop("prognosis", axis=1)

y = df["prognosis"]

print("\nNumber of Symptoms :", X.shape[1])
print("Number of Diseases :", y.nunique())

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape)
print("Testing Samples  :", X_test.shape)

# ============================================================
# MODEL
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Completed!")

# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy : {:.2f}%".format(accuracy*100))

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix Shape :", cm.shape)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Symptom":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

print("\nTop 20 Important Symptoms\n")

print(importance.head(20))

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model,"models/disease_model.pkl")

joblib.dump(list(X.columns),"models/features.pkl")

print("\nModel Saved Successfully!")

# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance.to_csv(

    "models/feature_importance.csv",

    index=False

)

print("Feature Importance Saved!")

print("\nProject Training Completed Successfully!")