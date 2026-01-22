import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from model import create_windows_features

# Parameters
RANDOM_STATE = 42
MODEL_PATH = "fusion_model.pkl"  # Change to evaluate other models

def evaluate_model(model, X, y):
    print("Model Evaluation Results")
    print("=" * 50)
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Overall Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    cm = plot_confusion_matrix(y, y_pred)
    
    return {'accuracy': accuracy, 'confusion_matrix': cm}

def determine_model_type(model_path):
    if "emg" in model_path.lower():
        return "emg"
    elif "imu" in model_path.lower():
        return "imu"
    elif "fusion" in model_path.lower():
        return "fusion"
    else:
        print(f"Warning: Cannot determine model type from {model_path}, defaulting to fusion")
        return "fusion"

def plot_confusion_matrix(actual_gestures, predicted_gestures, save_path="confusion_matrix.png"):
    confusion_results = confusion_matrix(actual_gestures, predicted_gestures)
    labels = sorted(set(actual_gestures) | set(predicted_gestures))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_results, cmap='Blues')
    plt.colorbar()
    
    for i in range(confusion_results.shape[0]):
        for j in range(confusion_results.shape[1]):
            plt.text(j, i, str(confusion_results[i,j]), ha='center', va='center', 
                    color='white' if confusion_results[i,j] > confusion_results.max()/2 else 'black')
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")
    
    return confusion_results

if __name__ == "__main__":
    
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded {MODEL_PATH}")
    
    model_type = determine_model_type(MODEL_PATH)
    print(f"Detected model type is {model_type}")
    
    df = pd.read_csv("filtered_all.csv")
    # print(f"Data loaded {len(df)} rows")
    
    X, y, feature_names = create_windows_features(df, feature_type=model_type, include_labels=True)
    
    results = evaluate_model(model, X, y)