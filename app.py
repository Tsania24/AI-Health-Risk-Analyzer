from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
import pandas as pd
import os
import io
import re
import base64
from PyPDF2 import PdfReader
import matplotlib
matplotlib.use('Agg') # Backend non-GUI agar tidak error di server
import matplotlib.pyplot as plt

app = Flask(__name__)

# Konfigurasi Upload
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'model_files'
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

# --- LOAD MODELS ---
def load_pickle(filename):
    try:
        path = os.path.join(MODEL_FOLDER, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load Diabetes
model_diabetes = load_pickle('model_diabetes.pkl')
diabetes_means = load_pickle('diabetes_means.pkl')

# Load Heart
model_heart = load_pickle('model_heart.pkl')
scaler_heart = load_pickle('scaler_heart.pkl')
heart_defaults = load_pickle('heart_defaults.pkl')

# Load Stroke
model_stroke = load_pickle('model_stroke.pkl')
stroke_encoders = load_pickle('stroke_encoders.pkl')

# --- PARSER PDF (REGEX KHUSUS PARAHITA) ---
def parse_medical_text(text):
    extracted = {}
    if not text: return extracted
    
    lines = text.split('\n')
    full_text = " ".join(lines)
    
    # 1. Identifikasi Dasar (Umur)
    age_match = re.search(r'(?:Umur|Age)\s*[:]?\s*.*?(\d{1,3})\s*Thn', full_text, re.IGNORECASE)
    if age_match: extracted['Age'] = int(age_match.group(1))

    # 2. Glukosa
    glucose_match = re.search(r'Glukosa.*?(?:Puasa|Sewaktu|2 Jam PP)?.*?(\d{2,3})', full_text, re.IGNORECASE)
    if glucose_match: extracted['Glucose'] = float(glucose_match.group(1))

    # 3. Kolesterol
    chol_match = re.search(r'(?:Kolesterol Total|Cholesterol|Kolesterol HDL).*?(\d{2,3})', full_text, re.IGNORECASE)
    if chol_match: extracted['Cholesterol'] = float(chol_match.group(1))

    # 4. Tensi / Blood Pressure
    bp_match = re.search(r'(?:Tensi|Tekanan Darah).*?(\d{2,3})\s*[\/]\s*(\d{2,3})', full_text, re.IGNORECASE)
    if bp_match:
        extracted['BloodPressure'] = float(bp_match.group(1)) 

    # 5. BMI
    bmi_match = re.search(r'BMI.*?(\d{2}(?:[.,]\d+)?)', full_text, re.IGNORECASE)
    if bmi_match: extracted['BMI'] = float(bmi_match.group(1).replace(',', '.'))
    
    # 6. HbA1c
    hba1c_match = re.search(r'HbA1c.*?(\d{1,2}[.,]\d{1,2})', full_text, re.IGNORECASE)
    if hba1c_match: extracted['HbA1c'] = hba1c_match.group(1).replace(',', '.')

    return extracted

# --- FUNGSI GENERATE PLOT (DARI CODE LAMA ANDA) ---
def generate_plots(input_data, risk_probs):
    # 1. Bar Chart: Perbandingan Data Klinis vs Rata-rata Normal
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Data Pasien vs Batas Normal (Contoh sederhana)
    params = ['Glukosa', 'Tensi', 'BMI', 'Kolesterol']
    patient_vals = [
        input_data.get('Glucose', 0),
        input_data.get('BloodPressure', 0),
        input_data.get('BMI', 0),
        input_data.get('Cholesterol', 0)
    ]
    normal_vals = [100, 120, 24, 200] # Batas normal umum
    
    x = np.arange(len(params))
    width = 0.35
    
    ax.bar(x - width/2, patient_vals, width, label='Pasien', color='#00B14F')
    ax.bar(x + width/2, normal_vals, width, label='Batas Normal', color='#e9ecef')
    
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.legend()
    ax.set_title('Profil Klinis Pasien')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    bar_chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # 2. Pie Chart: Distribusi Risiko 3 Penyakit
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    labels = ['Diabetes', 'Jantung', 'Stroke', 'Sehat']
    
    # Ambil probabilitas tertinggi sebagai risiko dominan
    max_risk = max(risk_probs['d'], risk_probs['h'], risk_probs['s'])
    health_score = 1 - max_risk
    
    sizes = [risk_probs['d'], risk_probs['h'], risk_probs['s'], health_score]
    colors = ['#ffc107', '#dc3545', '#fd7e14', '#28a745'] # Kuning, Merah, Orange, Hijau
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal') 
    ax2.set_title('Peta Risiko Kesehatan AI')

    img2 = io.BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    pie_chart_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    return bar_chart_url, pie_chart_url

# --- ROUTES ---

@app.route('/')
def home():
    # Ambil statistik sederhana untuk ditampilkan di placeholder form (opsional)
    stats = {}
    if diabetes_means:
        stats['BMI'] = {'mean': f"{diabetes_means.get('BMI', 0):.1f}", 'max': 67}
        stats['Glucose'] = {'mean': f"{diabetes_means.get('Glucose', 0):.1f}", 'max': 199}
    return render_template('index.html', stats=stats)

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'medical_record' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['medical_record']
    password = request.form.get('pdf_password', None)
    
    try:
        # FIX UTAMA: Gunakan BytesIO
        file_stream = io.BytesIO(file.read())
        pdf_reader = PdfReader(file_stream)
        
        if pdf_reader.is_encrypted:
            if not password:
                return jsonify({'status': 'encrypted_needed', 'message': 'File terkunci password.'})
            if not pdf_reader.decrypt(password):
                return jsonify({'status': 'encrypted_error', 'message': 'Password salah.'})

        text_content = ""
        for p in pdf_reader.pages:
            text = p.extract_text()
            if text: text_content += text + "\n"
        
        extracted = parse_medical_text(text_content)
        
        return jsonify({
            'status': 'success',
            'message': f'Berhasil scan. Ditemukan: {", ".join(extracted.keys())}',
            'extracted_data': extracted
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        # Helper konversi
        def get_val(key, default=0.0):
            try:
                return float(data.get(key, default))
            except:
                return default

        # Ambil Data Input
        age = get_val('Age', 30)
        bmi = get_val('BMI', 0)
        # Hitung BMI manual jika 0
        if bmi == 0 and get_val('height') > 0 and get_val('weight') > 0:
            bmi = get_val('weight') / ((get_val('height')/100)**2)
        
        glucose = get_val('Glucose', 100)
        bp = get_val('BloodPressure', 120)
        chol = get_val('Cholesterol', 200)

        # Variabel Probabilitas
        prob_d = 0
        prob_h = 0
        prob_s = 0

        results = {}

        # 1. PREDIKSI DIABETES
        if model_diabetes:
            d_input = [
                get_val('Pregnancies', 0),
                glucose,
                bp,
                get_val('SkinThickness', diabetes_means.get('SkinThickness', 20)),
                get_val('Insulin', diabetes_means.get('Insulin', 79)),
                bmi if bmi > 0 else diabetes_means.get('BMI', 31),
                get_val('DiabetesPedigreeFunction', 0.47),
                age
            ]
            prob_d = model_diabetes.predict_proba([d_input])[0][1]
            results['diabetes'] = {
                'risk': f"{prob_d*100:.1f}%",
                'label': 'Risiko Tinggi' if prob_d > 0.5 else 'Risiko Rendah',
                'color': 'red' if prob_d > 0.5 else '#00B14F'
            }

        # 2. PREDIKSI JANTUNG
        if model_heart:
            sex_val = 1 if data.get('gender', 'Male') == 'Male' else 0
            fbs_val = 1 if glucose > 120 else 0
            
            h_input = []
            features_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            
            for f in features_order:
                if f == 'age': val = age
                elif f == 'sex': val = sex_val
                elif f == 'trestbps': val = bp
                elif f == 'chol': val = chol
                elif f == 'fbs': val = fbs_val
                else: val = heart_defaults.get(f, 0)
                h_input.append(val)
            
            h_input_scaled = scaler_heart.transform([h_input])
            prob_h = model_heart.predict_proba(h_input_scaled)[0][1]
            results['heart'] = {
                'risk': f"{prob_h*100:.1f}%",
                'label': 'Terdeteksi Masalah' if prob_h > 0.5 else 'Normal',
                'color': 'red' if prob_h > 0.5 else '#00B14F'
            }

        # 3. PREDIKSI STROKE
        if model_stroke:
            s_input = [
                0 if data.get('gender') == 'Female' else 1,
                age,
                1 if bp > 140 else 0, # hypertension
                0, # heart_disease default
                1, # ever_married default
                2, # work_type default
                1, # Residence_type default
                glucose,
                bmi if bmi > 0 else 28.0,
                1  # smoking default
            ]
            prob_s = model_stroke.predict_proba([s_input])[0][1]
            results['stroke'] = {
                'risk': f"{prob_s*100:.1f}%",
                'label': 'Risiko Tinggi' if prob_s > 0.5 else 'Rendah',
                'color': 'red' if prob_s > 0.5 else '#00B14F'
            }

        # --- GENERATE PLOTS ---
        # Kita masukkan data real untuk grafik
        chart_inputs = {
            'Glucose': glucose,
            'BloodPressure': bp,
            'BMI': bmi,
            'Cholesterol': chol
        }
        risk_probs = {'d': prob_d, 'h': prob_h, 's': prob_s}
        
        bar_url, pie_url = generate_plots(chart_inputs, risk_probs)

        # Gabungkan hasil untuk dikirim ke JSON
        final_response = {
            **results, # Unpack diabetes, heart, stroke data
            'probability': f"{max(prob_d, prob_h, prob_s)*100:.1f}%", # Probabilitas tertinggi
            'kategori': 'PERLU PERHATIAN MEDIS' if max(prob_d, prob_h, prob_s) > 0.5 else 'SEHAT',
            'bmi_value': f"{bmi:.1f}",
            'rekomendasi': "Jaga pola makan dan rutin olahraga.",
            'clinical_bar_chart': bar_url,
            'pie_chart': pie_url
        }

        return jsonify(final_response)

    except Exception as e:
        print(f"Error Predict: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)