import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier

# Model Parameters
WINDOW_SIZE = 200
OVERLAP = 0.5
EMG_CHANNELS = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters
SVM_PARAM_GRID = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.01, 0.001],
    'svm__kernel': ['rbf']
}

LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'class_weight': "balanced",
    'random_state': RANDOM_STATE,
    'verbosity': -1 
}

def extract_features_window(window, column_names, sensor_type="fusion"):
    feats = []
    feat_names = []

    # EMG features descriptors
    if sensor_type in ["emg", "fusion"]:
        for i in range(min(EMG_CHANNELS, len(column_names))):
            signal = window[:, i]
            column = column_names[i]
            mav = np.mean(np.abs(signal))
            rms = np.sqrt(np.mean(signal**2))
            wl = np.sum(np.abs(np.diff(signal)))
            mean = np.mean(signal)
            std = np.std(signal)
            rng = np.max(signal) - np.min(signal)
            feats.extend([mav, rms, wl, mean, std, rng])
            feat_names.extend([f"{column}_MAV", f"{column}_RMS", f"{column}_WL",
                               f"{column}_Mean", f"{column}_Std", f"{column}_Range"])

    # IMU features descriptors
    if sensor_type in ["imu", "fusion"]:
        imu_start = EMG_CHANNELS
        while imu_start + 5 < len(column_names):
            acc_X, acc_Y, acc_Z = window[:, imu_start:imu_start+3].T
            acc_mean = np.mean([np.mean(acc_X), np.mean(acc_Y), np.mean(acc_Z)])
            acc_std = np.mean([np.std(acc_X), np.std(acc_Y), np.std(acc_Z)])
            acc_rms = np.mean([np.sqrt(np.mean(acc_X**2)),
                               np.sqrt(np.mean(acc_Y**2)),
                               np.sqrt(np.mean(acc_Z**2))])
            acc_rng = np.mean([np.max(acc_X)-np.min(acc_X),
                               np.max(acc_Y)-np.min(acc_Y),
                               np.max(acc_Z)-np.min(acc_Z)])
            acc_sma = (np.sum(np.abs(acc_X)) + np.sum(np.abs(acc_Y)) + np.sum(np.abs(acc_Z))) / len(acc_X)
            feats.extend([acc_mean, acc_std, acc_rms, acc_rng, acc_sma])
            feat_names.extend(["ACC_Mean", "ACC_Std", "ACC_RMS", "ACC_Range", "ACC_SMA"])

            gyro_X, gyro_Y, gyro_Z = window[:, imu_start+3:imu_start+6].T
            gyro_mean = np.mean([np.mean(gyro_X), np.mean(gyro_Y), np.mean(gyro_Z)])
            gyro_std = np.mean([np.std(gyro_X), np.std(gyro_Y), np.std(gyro_Z)])
            gyro_rms = np.mean([np.sqrt(np.mean(gyro_X**2)),
                                np.sqrt(np.mean(gyro_Y**2)),
                                np.sqrt(np.mean(gyro_Z**2))])
            gyro_rng = np.mean([np.max(gyro_X)-np.min(gyro_X),
                                np.max(gyro_Y)-np.min(gyro_Y),
                                np.max(gyro_Z)-np.min(gyro_Z)])
            gyro_sma = (np.sum(np.abs(gyro_X)) + np.sum(np.abs(gyro_Y)) + np.sum(np.abs(gyro_Z))) / len(gyro_X)
            feats.extend([gyro_mean, gyro_std, gyro_rms, gyro_rng, gyro_sma])
            feat_names.extend(["GYRO_Mean", "GYRO_Std", "GYRO_RMS", "GYRO_Range", "GYRO_SMA"])

            imu_start += 6

    return feats, feat_names

def create_windows_features(df, feature_type="fusion", include_labels=True):
    X, y = [], []
    feature_names = None
    step = int(WINDOW_SIZE * (1 - OVERLAP))
    df = df.drop(columns=["imu time"], errors='ignore')
    
    if include_labels:
        movement_col = "movement" if "movement" in df.columns else "out_movement"
        cols = df.columns.drop([movement_col])
        has_labels = True
    else:
        cols = df.columns
        has_labels = False

    for start in range(0, len(df) - WINDOW_SIZE, step):
        end = start + WINDOW_SIZE
        window = df.iloc[start:end][cols].values
        feats, feat_names = extract_features_window(window, cols, feature_type)
        X.append(feats)
        
        if has_labels:
            label = df.iloc[start:end][movement_col].mode()[0]
            y.append(label)
            
        if feature_names is None:
            feature_names = feat_names

    return np.array(X), np.array(y) if has_labels else None, feature_names

def create_ensemble_model():
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, class_weight="balanced"))
    ])
    
    lgbm = LGBMClassifier(**LGBM_PARAMS)
    
    ensemble = VotingClassifier(
        estimators=[("svm", svm_pipeline), ("lgbm", lgbm)], 
        voting="soft"
    )
    
    return ensemble, svm_pipeline, lgbm