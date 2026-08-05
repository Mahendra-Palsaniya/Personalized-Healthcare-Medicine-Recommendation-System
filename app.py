from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'healthai-secret-key-2026'
CORS(app)

# ============================================================================
# SIMPLE USER STORE (demo — replace with a real DB in production)
# ============================================================================
# Key: username, Value: dict with password and display name
USERS = {
    'demo': {'password': 'healthai123', 'name': 'Demo User'},
    'admin': {'password': 'admin123', 'name': 'Administrator'},
}

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

print("=" * 60)
print("HealthAI -- Loading models and datasets...")
print("=" * 60)

# --- Load SVC model ---
svc = None
try:
    with open(os.path.join(MODELS_DIR, 'svc_model.pkl'), 'rb') as f:
        svc = pickle.load(f)
    print("[OK] SVC Disease Prediction Model loaded")
except Exception as e:
    print("[WARN] SVC model not found:", e)

# --- Load Label Encoder ---
label_encoder = None
try:
    with open(os.path.join(MODELS_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    print("[OK] Label Encoder loaded")
except Exception as e:
    print("[WARN] Label Encoder not found:", e)

# --- Load Similarity Matrix for medicine recommendation ---
similarity = None
medicine_df = None
try:
    with open(os.path.join(MODELS_DIR, 'similarity.pkl'), 'rb') as f:
        similarity = pickle.load(f)
    medicine_df = pd.read_csv(os.path.join(DATASET_DIR, 'medicine.csv'))
    print("[OK] Medicine similarity matrix loaded")
except Exception as e:
    print("[WARN] Similarity/medicine data not found:", e)

# --- Load supplementary datasets ---
try:
    sym_des     = pd.read_csv(os.path.join(DATASET_DIR, 'symtoms_df.csv'))
    precautions = pd.read_csv(os.path.join(DATASET_DIR, 'precautions_df.csv'))
    workout     = pd.read_csv(os.path.join(DATASET_DIR, 'workout_df.csv'))
    description = pd.read_csv(os.path.join(DATASET_DIR, 'description.csv'))
    medications = pd.read_csv(os.path.join(DATASET_DIR, 'medications.csv'))
    diets       = pd.read_csv(os.path.join(DATASET_DIR, 'diets.csv'))
    print("[OK] Supplementary datasets loaded")
except Exception as e:
    print("[ERROR] Loading supplementary datasets:", e)
    sym_des = precautions = workout = description = medications = diets = None

print("=" * 60)
print("HealthAI Backend Ready!")
print("=" * 60)

# ============================================================================
# SYMPTOM & DISEASE MAPPINGS
# ============================================================================

symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14,
    'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17,
    'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23,
    'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
    'breathlessness': 27, 'sweating': 28, 'dehydration': 29,
    'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35,
    'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38,
    'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41,
    'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50,
    'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53,
    'runny_nose': 54, 'congestion': 55, 'chest_pain': 56,
    'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60,
    'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63,
    'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67,
    'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73,
    'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80,
    'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83,
    'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86,
    'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92,
    'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97,
    'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103,
    'increased_appetite': 104, 'polyuria': 105, 'family_history': 106,
    'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113,
    'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121,
    'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124,
    'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128,
    'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD',
    9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis',
    32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria',
    8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D',
    22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis',
    10: 'Common Cold', 34: 'Pneumonia',
    13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack',
    39: 'Varicose veins', 26: 'Hypothyroidism',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthristis', 5: 'Arthritis',
    0: '(vertigo) Paroymsal  Positional Vertigo',
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis',
    27: 'Impetigo'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def helper(dis):
    """Return description, precautions, medications, diet, workout for a disease."""
    desc = ""
    pre  = []
    med  = []
    die  = []
    wrkout = []

    if description is not None:
        d = description[description['Disease'] == dis]['Description']
        desc = " ".join([w for w in d])

    if precautions is not None:
        p = precautions[precautions['Disease'] == dis][
            ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [list(row) for row in p.values]

    if medications is not None:
        m = medications[medications['Disease'] == dis]['Medication']
        med = list(m.values)

    if diets is not None:
        di = diets[diets['Disease'] == dis]['Diet']
        die = list(di.values)

    if workout is not None:
        wk = workout[workout['disease'] == dis]['workout']
        wrkout = list(wk.values)

    return desc, pre, med, die, wrkout


def get_predicted_value(patient_symptoms):
    """Build feature vector and predict disease using SVC model."""
    if svc is None:
        return None, None
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        item = item.strip().lower().replace(' ', '_')
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    prediction = svc.predict([input_vector])[0]
    if label_encoder is not None:
        disease_name = label_encoder.inverse_transform([prediction])[0]
    else:
        disease_name = diseases_list.get(prediction, "Unknown")
    return disease_name, input_vector


def recommend_medicines(drug_name, n=5):
    """Return top-N similar medicines using cosine similarity."""
    if similarity is None or medicine_df is None:
        return []
    try:
        idx = medicine_df[
            medicine_df['Drug_Name'].str.lower() == drug_name.lower()
        ].index[0]
        distances = sorted(
            list(enumerate(similarity[idx])),
            reverse=True, key=lambda x: x[1]
        )
        recs = []
        for i, score in distances[1:n+1]:
            recs.append({
                'drug_name': medicine_df.iloc[i]['Drug_Name'],
                'reason': medicine_df.iloc[i]['Reason'],
                'score': round(float(score), 4)
            })
        return recs
    except Exception as e:
        print(f"Medicine recommendation error: {e}")
        return []


# ============================================================================
# ROUTES — PAGES
# ============================================================================

def login_required(f):
    """Decorator — redirect to /login if user is not in session."""
    from functools import wraps
    from urllib.parse import quote
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            msg = quote('Please sign in to access HealthAI.')
            return redirect(url_for('login') + f'?msg={msg}&type=info')
        return f(*args, **kwargs)
    return decorated


@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')


# ── Auth routes ─────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    from urllib.parse import quote
    # Already logged in → go home
    if 'username' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        action   = request.form.get('action', 'login')
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')

        if action == 'login':
            user = USERS.get(username)
            if user and user['password'] == password:
                session['username'] = username
                session['name']     = user['name']
                return redirect(url_for('home'))
            else:
                msg = quote('Incorrect username or password. Please try again.')
                return redirect(url_for('login') + f'?msg={msg}&type=error')

        elif action == 'register':
            if username in USERS:
                msg = quote('That username is already taken. Please choose another.')
                return redirect(url_for('login') + f'?msg={msg}&type=error#register')
            elif len(password) < 6:
                msg = quote('Password must be at least 6 characters.')
                return redirect(url_for('login') + f'?msg={msg}&type=error#register')
            else:
                fullname = request.form.get('fullname', username).strip()
                USERS[username] = {'password': password, 'name': fullname}
                session['username'] = username
                session['name']     = fullname
                return redirect(url_for('home'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    from urllib.parse import quote
    name = session.pop('name', 'User')
    session.pop('username', None)
    msg = quote(f'You have been signed out. See you next time, {name}!')
    return redirect(url_for('login') + f'?msg={msg}&type=info')


# ============================================================================
# ROUTES — API
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': svc is not None,
        'encoder_loaded': label_encoder is not None,
        'similarity_loaded': similarity is not None,
        'message': 'HealthAI backend is running!'
    })


@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return all symptom names for autocomplete."""
    symptom_list = [s.replace('_', ' ').title() for s in symptoms_dict.keys()]
    raw_list = list(symptoms_dict.keys())
    return jsonify({
        'symptoms': symptom_list,
        'raw': raw_list,
        'count': len(symptom_list)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disease from symptoms and return full recommendations.

    Request JSON:
        { "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"] }

    Response JSON:
        {
            "success": true,
            "disease": "Fungal infection",
            "description": "...",
            "precautions": [...],
            "medications": [...],
            "diets": [...],
            "workout": [...],
            "matched_symptoms": 3,
            "total_symptoms": 132
        }
    """
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'success': False, 'error': 'No symptoms provided'}), 400

        user_symptoms = data['symptoms']
        if not user_symptoms:
            return jsonify({'success': False, 'error': 'Symptoms list is empty'}), 400

        print("\nPrediction request:", user_symptoms)

        # Predict disease
        disease, input_vector = get_predicted_value(user_symptoms)
        if disease is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please run the notebook first to train and save models.'
            }), 503

        # Get recommendations
        desc, pre, med, die, wrkout = helper(disease)

        # Count matched symptoms
        matched = int(np.sum(input_vector)) if input_vector is not None else 0

        print("[OK] Predicted:", disease)

        return jsonify({
            'success': True,
            'disease': disease.strip(),
            'description': desc,
            'precautions': pre[0] if pre else [],
            'medications': med,
            'diets': die,
            'workout': list(wrkout),
            'matched_symptoms': matched,
            'total_symptoms': len(symptoms_dict)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/recommend_medicine', methods=['POST'])
def recommend_medicine():
    """
    Content-based medicine recommendation using cosine similarity.

    Request JSON:
        { "drug_name": "Metformin" }

    Response JSON:
        { "success": true, "recommendations": [...] }
    """
    try:
        data = request.get_json()
        if not data or 'drug_name' not in data:
            return jsonify({'success': False, 'error': 'No drug_name provided'}), 400

        drug_name = data['drug_name'].strip()
        n = int(data.get('n', 5))

        recommendations = recommend_medicines(drug_name, n)

        return jsonify({
            'success': True,
            'query': drug_name,
            'recommendations': recommendations,
            'count': len(recommendations)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/diseases', methods=['GET'])
def get_diseases():
    """Return the list of all 41 diseases covered."""
    return jsonify({
        'diseases': list(set(diseases_list.values())),
        'count': len(set(diseases_list.values()))
    })


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    
    print("HealthAI -- Personalized Healthcare & Medicine Recommendation System")
    
    print("API: http://localhost:5000")
    print("UI:  http://localhost:5000")
    
    app.run(host="0.0.0.0", port=5000)
