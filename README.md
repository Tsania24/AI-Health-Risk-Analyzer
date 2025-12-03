# 🏥 MediCare - AI Health Risk Analyzer

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Library](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![IoT](https://img.shields.io/badge/IoT-Bluetooth_Integration-0082FC?style=for-the-badge&logo=bluetooth&logoColor=white)
![Status](https://img.shields.io/badge/status-Final%20Project-success?style=for-the-badge)

> **Expert System & Machine Learning for Early Detection of Diabetes, Heart Disease, and Stroke Risks.**
> *Submitted for the Final Semester Exam (UAS) - Artificial Intelligence Course.*

---

## 📑 Table of Contents
- [About The Project](#-about-the-project)
- [Key Features](#-key-features)
- [Architecture & Technology](#-architecture--technology)
- [Datasets Used](#-datasets-used)
- [Installation & Usage](#-installation--usage)
- [Folder Structure](#-folder-structure)
- [Development Team](#-development-team)

---

## 📖 About The Project

**MediCare** is an intelligent web-based application designed to assist users in performing early health screening. The application utilizes the **Random Forest Classifier** algorithm to predict disease risk probabilities based on medical parameters (such as BMI, Glucose, Blood Pressure, etc.).

The system bridges the gap between hardware and software by featuring **IoT Integration** and **Smart PDF Parsing**. Users can automatically sync data from medical devices or upload lab results, eliminating the need for manual input and ensuring high data accuracy.

---

## 🚀 Key Features

### 1. 📡 IoT Smart Device Integration (Bluetooth)
Seamless hardware connectivity for real-time data synchronization.
* **Omron Digital Blood Pressure Monitor:** Automatically fetches Systolic/Diastolic blood pressure data via Bluetooth.
* **Accu-Chek Instant Blood Glucose Meter:** Syncs blood sugar levels directly into the web input fields.
* *Benefit:* Reduces human error in data entry and provides a seamless user experience.

### 2. 🔍 Multi-Disease Prediction
Risk analysis for 3 critical diseases in a single process:
* **Diabetes:** Predictions based on Glucose, BMI, Age, etc.
* **Heart Disease:** Analysis of Chest Pain, Cholesterol, Max Heart Rate.
* **Stroke:** Analysis of Smoking Status, Hypertension, and Medical History.

### 3. 📄 Smart PDF Extractor (OCR-like)
Advanced feature for reading digital medical records (PDF).
* Utilizes the `PyPDF2` library.
* Intelligent keyword search algorithm (Regex) to detect values: *Fasting Glucose, Total Cholesterol, Blood Pressure*.
* Supports Password Protected PDFs.

### 4. 📊 Interactive Data Visualization
Displays analysis results not just as text, but as visual charts:
* **Risk Chart:** Bar chart showing risk probability per disease.
* **Health Meter:** Visual indicator of health status (Healthy / Warning / Danger).
* **Recommendations:** Automated medical advice tailored to prediction results.

### 5. 🖨️ Generate PDF Report
Users can download the complete analysis result as a PDF document (`laporan_hasil_analisis.pdf`) containing the medical summary and risk charts for doctor consultation.

---

## 🛠 Architecture & Technology

This project is built using the following *Tech Stack*:

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Backend** | Python & Flask | Server-side logic and API routing (`app.py`) |
| **ML Model** | Scikit-Learn | Random Forest Algorithm & Preprocessing (`train_models.py`) |
| **IoT Connectivity** | **Web Bluetooth API** | Protocol to connect Omron & Accu-Chek devices to browser |
| **Data Processing** | Pandas & NumPy | CSV dataset manipulation and numerical calculations |
| **PDF Handling** | PyPDF2 | Text extraction from lab result PDF files |
| **Visualization** | Matplotlib | Generating static charts for reports |
| **Frontend** | HTML5, CSS3, JS | Responsive user interface (`style.css`) |

---

## 📂 Datasets Used

The Machine Learning models were trained using industry-standard public health datasets:

1.  **Diabetes:** *PIMA Indians Diabetes Database* (`diabetes.csv`)
2.  **Heart:** *Heart Disease UCI* (`heart.csv`)
3.  **Stroke:** *Stroke Prediction Dataset* (`healthcare-dataset-stroke-data.csv`)

---

## 💻 Installation & Usage

Follow these steps to run the project on your local machine:

### 1. Clone Repository
```bash
git clone [https://github.com/YOUR_USERNAME/medicare-ai-uas.git](https://github.com/YOUR_USERNAME/medicare-ai-uas.git)
cd medicare-ai-uas
```
### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install flask pandas numpy scikit-learn matplotlib pypdf2
```
### 4. Train Model (Required First)
```bash
python train_models.py
```
### 5. Run Application
```bash
python app.py
```
### 6. Acces Web
Open your browser and visit: 
```bash
http://127.0.0.1:5000/
```

---

## 📁 Folder Structure

Here is the overview of the project's file structure:
```bash
MediCare-Project/
├── app.py                   # Main Application (Flask Server & Logic)
├── train_models.py          # Script to Train & Save ML Models (Random Forest)
├── model_files/             # (Auto-Generated) Contains trained .pkl models
│   ├── model_diabetes.pkl
│   ├── model_heart.pkl
│   └── ...
├── static/                  # Static assets (CSS, Images)
│   └── style.css            # Custom styling for the UI
├── templates/               # HTML Templates
│   └── index.html           # Main Dashboard Interface
├── uploads/                 # Temporary folder for User's PDF uploads
├── diabetes.csv             # Dataset for Diabetes training
├── heart.csv                # Dataset for Heart Disease training
├── healthcare...csv         # Dataset for Stroke training
└── README.md                # Project Documentation
```

---

## 👥 Development Team

This project was developed by:
- Ayyub Valent Faturrahman (5323600032)
- Rissa Nur Azizah (5323600033)
- Tsania Zahira Soffa (5323600046)
- Rizky Nur Febianto D. (5323600058)

Major: Multimedia Engineering Technology
Course: Artificial Intelligence
