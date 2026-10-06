from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "Early Cancer Prediction - Project Report", border=0, align="C")
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("helvetica", "B", 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, border=0, align="L", fill=True)
        self.ln(12)

    def chapter_body(self, body):
        self.set_font("helvetica", "", 11)
        self.multi_cell(0, 6, body)
        self.ln()

    def add_bullet_point(self, text):
        self.set_font("helvetica", "", 11)
        self.cell(10, 6, "-", align="R")
        self.multi_cell(0, 6, text)
        self.ln(2)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# 1. Project Overview
pdf.chapter_title("1. Project Overview")
overview_text = (
    "The 'Early Cancer Prediction' project is an intelligent healthcare application designed "
    "to detect three prominent types of cancer: Brain Tumors, Oral Cancer, and Cervical Cancer "
    "at their early stages. The system leverages Machine Learning and Deep Learning models "
    "developed using comprehensive medical datasets."
)
pdf.chapter_body(overview_text)

# 2. How the Application is Running
pdf.chapter_title("2. How the Application is Running")
arch_text = (
    "The application relies on a robust architecture to serve predictions:\n\n"
    "Backend Framework: The backend is built on a Flask web server (app.py). It acts as the bridge "
    "between the user interface and the machine learning models.\n\n"
    "Model Loading: Pre-trained models (serialized using joblib) are stored in the 'Models/' directory. "
    "These models are loaded into memory when the server starts up, allowing for fast, real-time predictions.\n\n"
    "RESTful APIs: The server exposes several API endpoints such as:\n"
    "  - /api/predict/brain-tumor\n"
    "  - /api/predict/oral-cancer\n"
    "  - /api/predict/cervical-cancer\n\n"
    "Preprocessing & Validation: For image-based models (Brain and Oral), the app includes image "
    "validation logic to ensure users upload the correct type of image (e.g., verifying grayscale MRIs "
    "versus color photos). Images are then preprocessed (resized to 64x64, padded, and normalized) "
    "before being fed to the models. Cervical cancer data uses a pre-fitted StandardScaler."
)
pdf.chapter_body(arch_text)

# 3. Things You Need to Know About the Models
pdf.chapter_title("3. Model Specifications & Details")
pdf.chapter_body("Below is a breakdown of the models, their underlying algorithms, and their performance metrics:")

# Brain Tumor Model
pdf.set_font("helvetica", "B", 11)
pdf.cell(0, 8, "A. Brain Tumor Prediction Model")
pdf.ln(8)
pdf.add_bullet_point("Algorithm: Random Forest Ensemble utilizing CNN-extracted features.")
pdf.add_bullet_point("Input Data: Grayscale Brain MRI Scans (preprocessed to 64x64 array).")
pdf.add_bullet_point("Classes Detected: Glioma, Meningioma, Pituitary, and No Tumor (Healthy).")
pdf.add_bullet_point("Performance Metrics: Accuracy - 95.8%, F1 Score - 0.957, Precision - 0.960.")
pdf.add_bullet_point("Key Detail: The application actively validates uploaded images to ensure they meet the color and contrast profiles of an actual Brain MRI, preventing false readings from standard photos.")
pdf.ln(5)

# Oral Cancer Model
pdf.set_font("helvetica", "B", 11)
pdf.cell(0, 8, "B. Oral Cancer Prediction Model")
pdf.ln(8)
pdf.add_bullet_point("Algorithm: Ensemble Voting Mechanism combining Random Forest, Support Vector Machine (SVM), and Decision Tree classifiers.")
pdf.add_bullet_point("Input Data: Clinical images of oral mucosa and lesions.")
pdf.add_bullet_point("Classes Detected: Cancerous, Non-Cancerous.")
pdf.add_bullet_point("Performance Metrics: Accuracy - 93.4%, F1 Score - 0.931, Precision - 0.940.")
pdf.add_bullet_point("Key Detail: The application combines predictions from all three models using a majority voting system to determine the final prediction, ensuring high robustness.")
pdf.ln(5)

# Cervical Cancer Model
pdf.set_font("helvetica", "B", 11)
pdf.cell(0, 8, "C. Cervical Cancer Prediction Model")
pdf.ln(8)
pdf.add_bullet_point("Algorithm: Soft Voting Ensemble (combining Random Forest, AdaBoost, SVM, K-Nearest Neighbors, and Decision Tree).")
pdf.add_bullet_point("Input Data: 33 Tabular risk factors including Age, Number of sexual partners, Smoking habits, and STD/HPV diagnosis history.")
pdf.add_bullet_point("Performance Metrics: Accuracy - 96.5%, F1 Score - 0.962, Precision - 0.970.")
pdf.add_bullet_point("Key Detail: Alongside the ML model's prediction, the app calculates a 'Clinical Score' based on weighted high-risk factors. These scores are blended to provide a comprehensive Risk Percentage to the patient.")

# Export
pdf.output("Project_Report.pdf")
print("PDF created successfully!")
