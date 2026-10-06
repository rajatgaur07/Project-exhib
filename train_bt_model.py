import os
import glob
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

def train_brain_tumor_model():
    print("Training Full Brain Tumor Classifier Pipeline...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "Dataset", "Brain-tumor", "Training")
    test_dir = os.path.join(base_dir, "Dataset", "Brain-tumor", "Testing")
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    
    def load_dataset(dir_path):
        X, y = [], []
        for idx, cls_name in enumerate(classes):
            cls_path = os.path.join(dir_path, cls_name)
            if not os.path.exists(cls_path):
                continue
            files = glob.glob(os.path.join(cls_path, "*.jpg")) + glob.glob(os.path.join(cls_path, "*.jpeg")) + glob.glob(os.path.join(cls_path, "*.png"))
            for f in files:
                try:
                    img = Image.open(f).convert("L").resize((64, 64))
                    arr = np.array(img, dtype=np.float32) / 255.0
                    X.append(arr.flatten())
                    y.append(idx)
                except Exception:
                    pass
        return np.array(X), np.array(y)

    print("Loading Training set images...")
    X_train, y_train = load_dataset(train_dir)
    print("Loading Testing set images...")
    X_test, y_test = load_dataset(test_dir)
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=128, random_state=42),
        ExtraTreesClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
    )
    
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(f"Full Test Accuracy: {acc * 100:.2f}%")
    
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=classes))
    
    model_path = os.path.join(base_dir, "Models", "bt_rf_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved pipeline model to {model_path}")

if __name__ == "__main__":
    train_brain_tumor_model()
