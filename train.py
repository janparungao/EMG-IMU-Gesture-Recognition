import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier
import joblib
from model import *

def train_comparison_models(df):
    
    movement_col = "movement" if "movement" in df.columns else "out_movement"
    print(f"Label column {movement_col}")
    print(f"Gesture classes {sorted(df[movement_col].unique())}")
    
    print("EXTRACTING FEATURES FOR ALL MODELS")
    
    X_emg, y_emg, emg_feature_names = create_windows_features(df, feature_type="emg", include_labels=True)
    X_imu, y_imu, imu_feature_names = create_windows_features(df, feature_type="imu", include_labels=True)
    X_fusion, y_fusion, fusion_feature_names = create_windows_features(df, feature_type="fusion", include_labels=True)

    print("EMG only model trained")
    model_emg = train_single_model(X_emg, y_emg, "emg_model.pkl", "EMG-only")
    
    print("imu only model trained")
    model_imu = train_single_model(X_imu, y_imu, "imu_model.pkl", "IMU-only")
    
    print("fusion model trained")
    model_fusion = train_single_model(X_fusion, y_fusion, "fusion_model.pkl", "Fusion")
    
    print("Training completed, models saved as:")
    print("- emg_model.pkl")
    print("- imu_model.pkl") 
    print("- fusion_model.pkl")
    
    return model_emg, model_imu, model_fusion

def train_single_model(X, y, model_path, model_name):
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    ensemble, svm_pipeline, lgbm = create_ensemble_model()

    grid = GridSearchCV(svm_pipeline, SVM_PARAM_GRID, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_

    print(f"Training {model_name} LightGBM")
    lgbm.fit(X_train, y_train)

    print(f"Creating {model_name} ensemble")
    final_ensemble = VotingClassifier(
        estimators=[("svm", best_svm), ("lgbm", lgbm)], 
        voting="soft"
    )
    final_ensemble.fit(X_train, y_train)

    joblib.dump(final_ensemble, model_path)
    print(f"{model_name} model saved")

    return final_ensemble

if __name__ == "__main__":
    print("Training data loaded")
    df = pd.read_csv("filtered_all.csv")
    
    models = train_comparison_models(df)
    print("All models trained")