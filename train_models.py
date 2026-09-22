"""
Train IVF prediction models from the dataset.
Run this once: python train_models.py
Place IVF_Simulated_Dataset.xlsx in the same folder or update the path below.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle, os, json

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'IVF_Simulated_Dataset.xlsx')
MODELS_DIR = os.path.join(BASE, 'models')
DATA_DIR = os.path.join(BASE, 'data')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_excel(DATASET)
print(f"Loaded {len(df)} records")

features = ['Age','BMI','AMH','AFC','Basal_FSH','Basal_LH','Basal_E2',
            'Stimulation_Days','Dominant_Follicle_Size','E2_Trigger']
X = df[features]

# Encode Ovarian Response
le = LabelEncoder()
y_response = le.fit_transform(df['Ovarian_Response'])
y_ohss = df['OHSS_Risk']

X_train, X_test, yr_train, yr_test, yo_train, yo_test = train_test_split(
    X, y_response, y_ohss, test_size=0.2, random_state=42)

print("Training Ovarian Response model...")
rf_response = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf_response.fit(X_train, yr_train)
acc = accuracy_score(yr_test, rf_response.predict(X_test))
print(f"  Accuracy: {acc:.4f}")
print(classification_report(yr_test, rf_response.predict(X_test), target_names=le.classes_))

print("Training OHSS Risk model...")
rf_ohss = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf_ohss.fit(X_train, yo_train)
acc2 = accuracy_score(yo_test, rf_ohss.predict(X_test))
print(f"  Accuracy: {acc2:.4f}")

# Save models
pickle.dump(rf_response, open(f'{MODELS_DIR}/response_model.pkl','wb'))
pickle.dump(rf_ohss, open(f'{MODELS_DIR}/ohss_model.pkl','wb'))
pickle.dump(le, open(f'{MODELS_DIR}/label_encoder.pkl','wb'))
pickle.dump(features, open(f'{MODELS_DIR}/features.pkl','wb'))
feat_imp = dict(zip(features, rf_response.feature_importances_))
pickle.dump(feat_imp, open(f'{MODELS_DIR}/feature_importances.pkl','wb'))
print("Models saved!")

# Build patient dataset with IDs and names if patients.json doesn't exist
if not os.path.exists(f'{DATA_DIR}/patients.json'):
    names = [
        "Priya Sharma","Ananya Patel","Deepa Nair","Sunita Reddy","Kavitha Menon",
        "Rekha Iyer","Meena Gupta","Pooja Singh","Lakshmi Rao","Divya Kumar",
        "Nisha Pillai","Geeta Joshi","Shalini Bose","Ritu Agarwal","Anjali Verma",
        "Bharathi Krishnan","Uma Saxena","Shobha Desai","Padma Chatterjee","Asha Mehta",
        "Chitra Subramaniam","Latha Srinivasan","Malathi Venkat","Nirmala Balaji","Saritha Mohan",
        "Vani Natarajan","Hema Sundaram","Geetha Rajan","Kamala Balan","Savitha Murthy",
        "Radha Subramanian","Meenakshi Pillai","Sumathi Gopal","Nalini Chandran","Usha Naidu",
        "Pushpa Ramesh","Vijaya Suresh","Suganya Arjun","Mythili Prakash","Revathi Kannan",
        "Sudha Madhavan","Kalpana Raghavan","Sheela Nambiar","Saranya Balu","Ambika Seshadri",
        "Jaya Sundaresan","Mala Venkataraman","Gomathi Swaminathan","Vanitha Krishnamurthy","Thenmozhi Anand"
    ]
    df_s = df.head(50).copy()
    df_s['Patient_ID'] = [f'FERT{10001+i}' for i in range(50)]
    df_s['Patient_Name'] = names
    for c in ['Age','BMI','AMH','Basal_FSH','Basal_LH','Basal_E2','Dominant_Follicle_Size','E2_Trigger']:
        df_s[c] = df_s[c].round(2)
    df_s['AFC'] = df_s['AFC'].round(0).astype(int)
    df_s['Stimulation_Days'] = df_s['Stimulation_Days'].astype(int)
    df_s['Oocytes_Retrieved'] = df_s['Oocytes_Retrieved'].astype(int)
    cols = ['Patient_ID','Patient_Name','Age','BMI','AMH','AFC','Basal_FSH','Basal_LH','Basal_E2',
            'Stimulation_Days','Dominant_Follicle_Size','E2_Trigger','Oocytes_Retrieved',
            'Ovarian_Response','OHSS_Risk','Pregnancy']
    df_s[cols].to_json(f'{DATA_DIR}/patients.json', orient='records', indent=2)
    print(f"Patient dataset saved: 50 records")

print("\nAll done! Run: python app.py")
