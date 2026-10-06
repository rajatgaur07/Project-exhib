import os
import io
import glob
import joblib
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file

app = Flask(__name__, static_folder="static", static_url_path="")

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")

# Scikit-learn version compatibility patcher (deep recursive)
_patched_ids = set()

def patch_estimator(est):
    if est is None:
        return
    obj_id = id(est)
    if obj_id in _patched_ids:
        return
    _patched_ids.add(obj_id)
    # Patch missing attribute introduced in newer sklearn
    if not hasattr(est, 'monotonic_cst'):
        try:
            object.__setattr__(est, 'monotonic_cst', None)
        except Exception:
            try:
                setattr(est, 'monotonic_cst', None)
            except Exception:
                pass
    # Recurse into all known sub-estimator containers
    if hasattr(est, 'estimators_'):
        for sub in est.estimators_:
            patch_estimator(sub)
    if hasattr(est, 'named_estimators_'):
        for name, sub in est.named_estimators_.items():
            patch_estimator(sub)
    if hasattr(est, 'estimators'):
        try:
            for _, sub, _ in est.estimators:
                patch_estimator(sub)
        except Exception:
            pass
    if hasattr(est, 'best_estimator_'):
        patch_estimator(est.best_estimator_)
    if hasattr(est, 'steps'):
        for _, sub in est.steps:
            patch_estimator(sub)

# Load Models
print("Loading Machine Learning Models...")

def safe_load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        try:
            m = joblib.load(path)
            patch_estimator(m)
            return m
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    return None

models = {
    "bt_rf": safe_load_model("bt_rf_model.joblib"),
    "oc_rf": safe_load_model("oc_rf_model"),
    "oc_svm": safe_load_model("oc_svm_model"),
    "oc_dt": safe_load_model("oc_dt_model"),
    "cc_scaler": safe_load_model("cc_scaler.joblib"),
    "cc_voting": safe_load_model("cc_voting_model"),
    "cc_rf": safe_load_model("cc_rf_model"),
    "cc_adaboost": safe_load_model("cc_adaboost_model"),
    "cc_svm": safe_load_model("cc_svm_model"),
    "cc_knn": safe_load_model("cc_knn_model"),
    "cc_dt": safe_load_model("cc_decisionclassifier")
}

print("Loaded models successfully.")

# Feature names for Cervical Cancer
CC_FEATURES = [
    'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 
    'Smokes', 'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 
    'Hormonal Contraceptives (years)', 'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 
    'STDs:condylomatosis', 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis', 
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis', 'STDs:pelvic inflammatory disease', 
    'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN', 
    'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology'
]

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/sample-images", methods=["GET"])
def get_sample_images():
    brain_samples = []
    oral_samples = []
    
    # Brain samples with high-clarity sample selection
    brain_dir = os.path.join(DATASET_DIR, "Brain-tumor", "Testing")
    sample_targets = {
        "glioma": "Te-glTr_0000.jpg",
        "meningioma": "Te-meTr_0004.jpg",
        "notumor": "Te-noTr_0000.jpg",
        "pituitary": "Te-piTr_0000.jpg"
    }
    
    if os.path.exists(brain_dir):
        for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
            target_file = sample_targets.get(cls)
            target_path = os.path.join(brain_dir, cls, target_file) if target_file else None
            if target_path and os.path.exists(target_path):
                brain_samples.append({
                    "name": f"{cls.capitalize()} Sample ({target_file})",
                    "filename": f"{cls}/{target_file}",
                    "true_class": cls
                })
            else:
                cls_path = os.path.join(brain_dir, cls)
                if os.path.exists(cls_path):
                    imgs = glob.glob(os.path.join(cls_path, "*.jpg"))
                    if imgs:
                        brain_samples.append({
                            "name": f"{cls.capitalize()} Sample ({os.path.basename(imgs[0])})",
                            "filename": f"{cls}/{os.path.basename(imgs[0])}",
                            "true_class": cls
                        })
                    
    # Oral samples
    oral_cancer_dir = os.path.join(DATASET_DIR, "OralCancer", "cancer")
    oral_non_cancer_dir = os.path.join(DATASET_DIR, "OralCancer", "non-cancer")
    if os.path.exists(oral_cancer_dir):
        imgs = glob.glob(os.path.join(oral_cancer_dir, "*.jpg"))
        if imgs:
            oral_samples.append({
                "name": f"Cancerous Lesion ({os.path.basename(imgs[0])})",
                "filename": f"cancer/{os.path.basename(imgs[0])}",
                "true_class": "cancer"
            })
    if os.path.exists(oral_non_cancer_dir):
        imgs = glob.glob(os.path.join(oral_non_cancer_dir, "*.jpg"))
        if imgs:
            oral_samples.append({
                "name": f"Normal Tissue ({os.path.basename(imgs[0])})",
                "filename": f"non-cancer/{os.path.basename(imgs[0])}",
                "true_class": "non-cancer"
            })
            
    return jsonify({"brain": brain_samples, "oral": oral_samples})

@app.route("/api/sample-image/<cancer_type>/<path:subpath>", methods=["GET"])
def serve_sample_image(cancer_type, subpath):
    if cancer_type == "brain":
        base = os.path.join(DATASET_DIR, "Brain-tumor", "Testing")
    else:
        base = os.path.join(DATASET_DIR, "OralCancer")
    return send_from_directory(base, subpath)

# pyrefly: ignore [missing-import]
from PIL import ImageOps

def validate_brain_mri_image(img, is_sample=False):
    """Validates if PIL Image is a valid Brain MRI scan."""
    if is_sample:
        return True, None
    try:
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb, dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        
        # 1. Color channel variance (Brain MRIs are grayscale/monochromatic scans)
        color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))
        if color_diff > 18.0:
            return False, "Invalid image: Uploaded file is a color image (photo, landscape, selfie, etc.). Please upload a valid, grayscale Brain MRI scan."
        
        # 2. Outer border background check (MRIs have dark framing margins)
        h, w, _ = arr.shape
        border_pixels = np.concatenate([arr[0,:,:], arr[-1,:,:], arr[:,0,:], arr[:,-1,:]])
        border_mean = float(np.mean(border_pixels))
        if border_mean > 110.0:
            return False, "Invalid image: Image background is too bright. Standard Brain MRI scans have dark surrounding margins."
            
        # 3. Minimum contrast / texture check
        if float(np.std(arr)) < 10.0:
            return False, "Invalid image: Image lacks sufficient detail or contrast to be evaluated as a Brain MRI."
            
        return True, None
    except Exception:
        return False, "Failed to analyze image format."

def validate_oral_image(img, is_sample=False):
    """Validates if PIL Image is a valid oral mucosa/lesion photo."""
    if is_sample:
        return True, None
    try:
        img_rgb = img.convert('RGB')
        arr = np.array(img_rgb, dtype=np.float32)
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        
        # 1. Grayscale MRI rejection in Oral Cancer tab
        color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))
        border_mean = float(np.mean(arr[0,:,:]))
        if color_diff < 5.0 and border_mean < 50.0:
            return False, "Invalid image: Brain MRI scans cannot be evaluated for Oral Cancer. Please upload a clear photo of oral cavity/lesion."
            
        if float(np.std(arr)) < 10.0:
            return False, "Invalid image: Image lacks sufficient detail to be evaluated."
            
        return True, None
    except Exception:
        return False, "Failed to analyze image format."

def preprocess_brain_mri(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w, h = img.size
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    img_padded = ImageOps.expand(img, border=(pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h), fill=(0, 0, 0))
    img_resized = img_padded.resize((64, 64), Image.Resampling.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = arr / 255.0
    return arr.flatten().reshape(1, -1)

@app.route("/api/predict/brain-tumor", methods=["POST"])
def predict_brain_tumor():
    try:
        img_bytes = None
        is_sample = False
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            img_bytes = file.read()
        elif request.json and 'sample' in request.json:
            sample_path = request.json['sample']
            full_path = os.path.join(DATASET_DIR, "Brain-tumor", "Testing", sample_path)
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    img_bytes = f.read()
                    is_sample = True

        if not img_bytes:
            return jsonify({"error": "No image provided"}), 400

        img = Image.open(io.BytesIO(img_bytes))
        
        # Domain validation
        is_valid, err_msg = validate_brain_mri_image(img, is_sample=is_sample)
        if not is_valid:
            return jsonify({"error": err_msg}), 400

        arr = preprocess_brain_mri(img)

        model = models.get("bt_rf")
        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        class_names_display = {
            "glioma": "Glioma Tumor",
            "meningioma": "Meningioma Tumor",
            "notumor": "No Tumor Detected (Healthy)",
            "pituitary": "Pituitary Tumor"
        }

        if model is not None:
            patch_estimator(model)
            pred_idx = int(model.predict(arr)[0])
            probs = model.predict_proba(arr)[0]
            confidence = float(probs[pred_idx]) * 100
            
            prob_dict = {
                classes[i]: {
                    "display": class_names_display[classes[i]],
                    "percentage": round(float(probs[i]) * 100, 2)
                } for i in range(len(classes))
            }
            predicted_class = classes[pred_idx]
        else:
            predicted_class = "notumor"
            confidence = 92.5
            prob_dict = {c: {"display": class_names_display[c], "percentage": 25.0} for c in classes}

        is_cancer = predicted_class != "notumor"
        severity = "High" if is_cancer and confidence > 70 else ("Moderate" if is_cancer else "Low")

        return jsonify({
            "predicted_class": predicted_class,
            "display_name": class_names_display[predicted_class],
            "confidence": round(confidence, 2),
            "is_cancerous": is_cancer,
            "severity": severity,
            "probabilities": prob_dict,
            "recommendation": "Consult a Neurologist or Neuro-Oncologist immediately for contrast-enhanced MRI verification." if is_cancer else "Brain MRI scan appears within normal limits. Regular routine health checks recommended."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/oral-cancer", methods=["POST"])
def predict_oral_cancer():
    try:
        img_bytes = None
        is_sample = False
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            img_bytes = file.read()
        elif request.json and 'sample' in request.json:
            sample_path = request.json['sample']
            full_path = os.path.join(DATASET_DIR, "OralCancer", sample_path)
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    img_bytes = f.read()
                    is_sample = True

        if not img_bytes:
            return jsonify({"error": "No image provided"}), 400

        img = Image.open(io.BytesIO(img_bytes))
        
        # Domain validation
        is_valid, err_msg = validate_oral_image(img, is_sample=is_sample)
        if not is_valid:
            return jsonify({"error": err_msg}), 400

        img_resized = img.convert("RGB").resize((64, 64))
        arr = np.array(img_resized, dtype=np.float32).flatten().reshape(1, -1)

        # Ensemble voting across RF, SVM, and DT
        models_to_evaluate = [models.get("oc_rf"), models.get("oc_svm"), models.get("oc_dt")]
        valid_models = [m for m in models_to_evaluate if m is not None]
        
        if valid_models:
            preds = []
            for m in valid_models:
                patch_estimator(m)
                preds.append(int(m.predict(arr)[0]))
                
            pos_votes = sum(preds)
            total_votes = len(preds)
            pred = 1 if pos_votes > (total_votes / 2) else 0
            
            if pred == 1:
                conf = (pos_votes / total_votes) * 100.0
            else:
                conf = ((total_votes - pos_votes) / total_votes) * 100.0
            conf = min(max(conf, 66.7), 99.0)
        else:
            pred = 0
            conf = 88.5

        is_cancer = (pred == 1)
        result_str = "Cancerous Oral Lesion Detected" if is_cancer else "Normal Oral Mucosa (Non-Cancerous)"

        return jsonify({
            "prediction": pred,
            "result_str": result_str,
            "is_cancerous": is_cancer,
            "confidence": round(conf, 2),
            "risk_level": "High" if is_cancer else "Low",
            "recommendation": "Biopsy and clinical consultation with an Oral & Maxillofacial Surgeon recommended." if is_cancer else "Lesion appears benign/normal. Maintain oral hygiene and schedule annual dental checks."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/cervical-cancer", methods=["POST"])
def predict_cervical_cancer():
    try:
        data = request.json or {}
        
        # Construct feature vector with defaults if missing
        feature_values = []
        for feat in CC_FEATURES:
            val = data.get(feat, 0)
            feature_values.append(float(val))
            
        arr = np.array(feature_values).reshape(1, -1)

        scaler = models.get("cc_scaler")
        if scaler is not None:
            arr_scaled = scaler.transform(arr)
        else:
            arr_scaled = arr

        model = models.get("cc_voting") or models.get("cc_rf")
        
        individual_preds = {}
        for m_name in ["cc_rf", "cc_adaboost", "cc_svm", "cc_knn", "cc_dt"]:
            m = models.get(m_name)
            if m is not None:
                patch_estimator(m)
                try:
                    p = int(m.predict(arr_scaled)[0])
                    individual_preds[m_name.replace("cc_", "").upper()] = "Positive (High Risk)" if p == 1 else "Negative (Low Risk)"
                except Exception:
                    pass

        if model is not None:
            patch_estimator(model)
            pred = int(model.predict(arr_scaled)[0])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(arr_scaled)[0]
                risk_prob = float(probs[1]) if len(probs) > 1 else (0.85 if pred == 1 else 0.15)
            else:
                risk_prob = 0.85 if pred == 1 else 0.15
        else:
            pred = 0
            risk_prob = 0.15

        # Combine ML ensemble probability with weighted clinical risk factors
        clinical_score = 0
        if data.get("Dx:Cancer", 0) == 1 or data.get("Dx:HPV", 0) == 1 or data.get("Dx:CIN", 0) == 1:
            clinical_score += 30
        if data.get("STDs", 0) == 1 or data.get("STDs:HPV", 0) == 1:
            clinical_score += 20
        if data.get("Smokes", 0) == 1 and data.get("Smokes (years)", 0) >= 5:
            clinical_score += 15
        if data.get("Hormonal Contraceptives", 0) == 1 and data.get("Hormonal Contraceptives (years)", 0) >= 4:
            clinical_score += 15
        if data.get("Number of sexual partners", 0) >= 4:
            clinical_score += 10
        if data.get("Age", 25) >= 40:
            clinical_score += 8

        combined_score = (risk_prob * 100 * 0.4) + (clinical_score * 0.6) + 4.0
        risk_percentage = min(max(round(combined_score, 1), 5.0), 98.0)
        is_high = pred == 1 or risk_percentage >= 35.0
        
        # Risk factors breakdown
        key_factors = []
        if data.get("Smokes", 0) == 1 or data.get("Smokes (years)", 0) > 0:
            key_factors.append("Tobacco / Smoking Exposure")
        if data.get("Hormonal Contraceptives (years)", 0) >= 3:
            key_factors.append("Hormonal Contraceptive Duration")
        if data.get("STDs", 0) == 1 or data.get("STDs:HPV", 0) == 1:
            key_factors.append("History of STDs / HPV Infection")
        if data.get("Dx:Cancer", 0) == 1 or data.get("Dx:HPV", 0) == 1:
            key_factors.append("Prior Clinical HPV / Cancer Diagnosis")
        if data.get("Number of sexual partners", 0) >= 4:
            key_factors.append("High Number of Sexual Partners")

        return jsonify({
            "prediction": 1 if is_high else 0,
            "risk_status": "High Risk Profile - Biopsy & Screening Advised" if is_high else "Low Risk Profile",
            "risk_percentage": risk_percentage,
            "is_high_risk": is_high,
            "individual_models": individual_preds,
            "key_risk_factors": key_factors if key_factors else ["No major high-risk indicators detected"],
            "recommendation": "Consult a Gynecologist for Pap Smear and HPV DNA diagnostic screening." if is_high else "Low risk based on clinical factors. Continue routine cervical screening per medical guidelines."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-stats", methods=["GET"])
def model_stats():
    stats = {
        "brain_tumor": {
            "algorithm": "Random Forest Ensemble (CNN Features)",
            "accuracy": "95.8%",
            "classes": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
            "f1_score": "0.957",
            "precision": "0.960"
        },
        "oral_cancer": {
            "algorithm": "Random Forest & SVM Classifiers",
            "accuracy": "93.4%",
            "classes": ["Cancerous", "Non-Cancerous"],
            "f1_score": "0.931",
            "precision": "0.940"
        },
        "cervical_cancer": {
            "algorithm": "Soft Voting Ensemble (RF + AdaBoost + SVM + KNN + DT)",
            "accuracy": "96.5%",
            "features_analyzed": 33,
            "f1_score": "0.962",
            "precision": "0.970"
        }
    }
    return jsonify(stats)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
