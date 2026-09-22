# 🩺 AI-Based Clinical Decision Support System for IVF Ovarian Stimulation

An end-to-end Machine Learning solution designed to assist clinicians in optimizing In Vitro Fertilization (IVF) ovarian stimulation protocols through predictive modeling and Explainable AI (XAI)[cite: 2, 4].

---

## 🎯 Overview & Objectives

In Vitro Fertilization (IVF) treatments require precise monitoring of ovarian stimulation to optimize trigger timing and prevent dangerous complications such as Ovarian Hyperstimulation Syndrome (OHSS)[cite: 2].

This decision support system provides data-driven clinical guidance across two key tasks:
1. **Optimal Trigger Timing Prediction:** Recommending the precise timing for triggering ovulation based on longitudinal clinical response[cite: 2].
2. **OHSS Risk Classification:** Early detection and classification of hyperstimulation risks[cite: 2].

---

## 📈 Model Performance & Features

- **Clinical Dataset:** Built and validated on **~6,000–7,000 longitudinal clinical records** (`IVF_Dataset.xlsx`)[cite: 2, 4].
- **Predictive Accuracy:** Achieved **~90% classification accuracy** using time-series feature engineering and rigorous cross-validation[cite: 2].
- **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** to generate feature importance plots and patient-level predictions, ensuring model decisions are interpretable and trustworthy for medical professionals[cite: 2].
- **Interactive Dashboard:** Visualized patient risk profiles, feature contributions, and recommendations via a user-friendly web interface[cite: 2, 4].

---

## 🛠️ Tech Stack & Tools

- **Language:** Python[cite: 2, 4]
- **Machine Learning:** Scikit-learn, SHAP (Explainable AI), Cross-Validation[cite: 2]
- **Data Engineering:** Pandas, NumPy, Time-Series Feature Extraction[cite: 2]
- **Web Application:** Flask, HTML5 Templates[cite: 2, 4]
- **Data Format:** Microsoft Excel (`IVF_Dataset.xlsx`)[cite: 4]

---

## 📁 Repository Structure

```text
├── data/                       # Preprocessing scripts and data utilities
├── models/                     # Saved ML model checkpoints
├── reports/                    # Model evaluation outputs & SHAP plots
├── templates/                  # Frontend HTML interfaces for clinical workflow
├── uploads/                    # Temporary storage for batch patient file processing
├── app.py                      # Flask web server and routing application
├── ferticare.txt               # System notes & clinical specifications
├── IVF_Dataset.xlsx            # Source clinical dataset (~1.08 MB)
├── requirements.txt            # Python environment packages
├── train_models.py             # Feature engineering & ML training script
└── README.md                   # Documentation
