import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

if not os.path.exists('model_files'):
    os.makedirs('model_files')

# ==========================================
# 1. LATIH MODEL DIABETES
# ==========================================
print("Melatih Model Diabetes...")
df_diabetes = pd.read_csv('diabetes.csv')
X_d = df_diabetes.drop('Outcome', axis=1)
y_d = df_diabetes['Outcome']

# Imputasi nilai 0 dengan mean (kecuali kehamilan)
cols_d = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer_d = SimpleImputer(missing_values=0, strategy='mean')
X_d[cols_d] = imputer_d.fit_transform(X_d[cols_d])

model_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
model_diabetes.fit(X_d, y_d)

with open('model_files/model_diabetes.pkl', 'wb') as f:
    pickle.dump(model_diabetes, f)
    
# Simpan rata-rata untuk imputasi nanti di app.py
diabetes_means = X_d.mean().to_dict()
with open('model_files/diabetes_means.pkl', 'wb') as f:
    pickle.dump(diabetes_means, f)

# ==========================================
# 2. LATIH MODEL JANTUNG (HEART)
# ==========================================
print("Melatih Model Jantung...")
df_heart = pd.read_csv('heart.csv')

# Perbaikan nama kolom: Dataset Anda menggunakan 'target', bukan 'output'
X_h = df_heart.drop('target', axis=1)
y_h = df_heart['target']

# Scaling
scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)

model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_h_scaled, y_h)

with open('model_files/model_heart.pkl', 'wb') as f:
    pickle.dump(model_heart, f)
with open('model_files/scaler_heart.pkl', 'wb') as f:
    pickle.dump(scaler_h, f)
    
# Simpan nilai rata-rata kolom untuk default value
heart_defaults = X_h.mean().to_dict()
with open('model_files/heart_defaults.pkl', 'wb') as f:
    pickle.dump(heart_defaults, f)

# ==========================================
# 3. LATIH MODEL STROKE
# ==========================================
print("Melatih Model Stroke...")
df_stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Preprocessing
df_stroke['bmi'] = df_stroke['bmi'].fillna(df_stroke['bmi'].mean())
df_stroke = df_stroke.drop(['id'], axis=1)

# Encoding Categorical
le_gender = LabelEncoder()
df_stroke['gender'] = le_gender.fit_transform(df_stroke['gender'])

le_married = LabelEncoder()
df_stroke['ever_married'] = le_married.fit_transform(df_stroke['ever_married'])

le_work = LabelEncoder()
df_stroke['work_type'] = le_work.fit_transform(df_stroke['work_type'])

le_residence = LabelEncoder()
df_stroke['Residence_type'] = le_residence.fit_transform(df_stroke['Residence_type'])

le_smoking = LabelEncoder()
df_stroke['smoking_status'] = le_smoking.fit_transform(df_stroke['smoking_status'])

X_s = df_stroke.drop('stroke', axis=1)
y_s = df_stroke['stroke']

# Handle imbalanced data
model_stroke = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_stroke.fit(X_s, y_s)

with open('model_files/model_stroke.pkl', 'wb') as f:
    pickle.dump(model_stroke, f)
    
# Simpan encoders
encoders = {
    'gender': le_gender,
    'ever_married': le_married,
    'work_type': le_work,
    'Residence_type': le_residence,
    'smoking_status': le_smoking
}
with open('model_files/stroke_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("SUKSES! Semua model telah dilatih dan disimpan di folder model_files/.")