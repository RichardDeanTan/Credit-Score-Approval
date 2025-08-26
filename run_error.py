import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import os

def recreate_random_forest():
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=7,
        min_samples_leaf=3,
        max_features=None,
        random_state=42,
        n_jobs=-1
    )
    
    df_train = pd.read_csv('clean data/train_transformed_upsampled.csv')
    X_train = df_train.drop('Credit_Score', axis=1)
    y_train = df_train['Credit_Score']
    rf_model.fit(X_train, y_train)
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(rf_model, 'model/rf_optuna_model_fixed.joblib')

def test_loading():
    df_test = pd.read_csv('clean data/test_transformed.csv')
    X_test = df_test.drop('Credit_Score', axis=1)
    y_test = df_test['Credit_Score']

    try:
        rf_model = joblib.load('model/rf_optuna_model_fixed.joblib')
        print("✅ Random Forest model loaded successfully!")
        
        y_pred = rf_model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"✅ Test Accuracy: {acc:.4f}")
        print(f"✅ Test F1-score (weighted): {f1:.4f}")
    except Exception as e:
        print(f"❌ Random Forest loading failed: {e}")

if __name__ == "__main__":    
    recreate_random_forest()
    test_loading()