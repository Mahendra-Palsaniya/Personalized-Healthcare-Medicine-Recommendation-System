"""
Train all ML models and save pickle files.
Run this script from the HealthAI directory to generate model files.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from warnings import filterwarnings
filterwarnings('ignore')

# Paths
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 55)
print("HealthAI -- Model Training Script")
print("=" * 55)

# --- 1. Load Training Data ---
print("\n[1] Loading Training.csv ...")
dataset = pd.read_csv(os.path.join(DATASET_DIR, 'Training.csv'))
print("    Shape:", dataset.shape)

# --- 2. Preprocess ---
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
print("    Train:", X_train.shape[0], "| Test:", X_test.shape[0])

# --- 3. Train Models ---
models = {
    'SVC':              SVC(kernel='linear'),
    'RandomForest':     RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

trained = {}
print("\n[2] Training models:")
for name, model in models.items():
    print("    Training", name, "...", end=' ', flush=True)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    trained[name] = model
    print("OK  Accuracy:", round(acc * 100, 2), "%")

# --- 4. Save Models ---
print("\n[3] Saving models...")
with open(os.path.join(MODELS_DIR, 'svc_model.pkl'), 'wb') as f:
    pickle.dump(trained['SVC'], f)
with open(os.path.join(MODELS_DIR, 'rf_model.pkl'), 'wb') as f:
    pickle.dump(trained['RandomForest'], f)
with open(os.path.join(MODELS_DIR, 'gb_model.pkl'), 'wb') as f:
    pickle.dump(trained['GradientBoosting'], f)
with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
print("    Saved: svc_model.pkl, rf_model.pkl, gb_model.pkl, label_encoder.pkl")

# --- 5. Content-Based Medicine Recommendation ---
print("\n[4] Building medicine recommendation (TF-IDF + Cosine)...")
medicine_path = os.path.join(DATASET_DIR, 'medicine.csv')
medications   = pd.read_csv(os.path.join(DATASET_DIR, 'medications.csv'))

if os.path.exists(medicine_path):
    medicine_df = pd.read_csv(medicine_path)
    print("    Loaded medicine.csv:", len(medicine_df), "drugs")
else:
    med_data = []
    for _, row in medications.iterrows():
        disease  = row['Disease']
        meds_str = row['Medication']
        try:
            meds_list = eval(meds_str) if isinstance(meds_str, str) else [meds_str]
        except Exception:
            meds_list = [str(meds_str)]
        for m in meds_list:
            med_data.append({
                'Drug_Name':   m.strip(),
                'Reason':      disease,
                'Description': 'Used to treat ' + disease
            })
    medicine_df = pd.DataFrame(med_data).drop_duplicates().reset_index(drop=True)
    medicine_df.to_csv(medicine_path, index=False)
    print("    Created medicine.csv:", len(medicine_df), "drugs")

medicine_df['tags'] = (
    medicine_df['Description'].fillna('') + ' ' +
    medicine_df['Reason'].fillna('')
).str.lower()

vectorizer   = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(medicine_df['tags'])
similarity   = cosine_similarity(tfidf_matrix)
print("    Similarity matrix:", similarity.shape)

with open(os.path.join(MODELS_DIR, 'similarity.pkl'), 'wb') as f:
    pickle.dump(similarity, f)
print("    Saved: similarity.pkl")

# --- 6. Summary ---
print("\n" + "=" * 55)
print("SAVED MODEL FILES:")
print("=" * 55)
for f in sorted(os.listdir(MODELS_DIR)):
    size = os.path.getsize(os.path.join(MODELS_DIR, f))
    print(f"   {f:30s}  {size/1024:7.1f} KB")

print("\nTraining complete! Run:  python app.py")
print("=" * 55)
