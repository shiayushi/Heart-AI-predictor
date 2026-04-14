# Heart-AI-predictior

A Machine Learning web application built with **Streamlit** that predicts the risk of heart disease based on patient medical parameters. The app also generates visual explanations using **SHAP** and downloadable PDF reports.

---

## 🚀 Live Demo
If deployed on Streamlit Cloud:


---

## 📌 Features

- 🧠 ML-based heart disease prediction
- 📊 SHAP explainability (feature impact visualization)
- 🧾 PDF report generation with patient details
- 👤 Patient name + contact storage
- 📉 Risk classification (Low / Medium / High)
- 🎨 Simple and interactive UI (Streamlit)

---

## 🛠️ Tech Stack

- Python 🐍
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Matplotlib
- SHAP
- ReportLab (PDF generation)

---

## 📂 Project Structure
heart-disease-app/
│
├── app.py # Main Streamlit app
├── model.pkl # Trained ML model
├── scaler.pkl 
├── requirements.txt # Dependencies
├── README.md # Project documentation

## ⚙️ Installation (Local Setup)

```bash
# Clone repository
git clone https://github.com/your-username/heart-disease-app.git

# Move into directory
cd heart-disease-app

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py

📊 Model Information
Algorithm: (e.g., Logistic Regression / Random Forest)
Input Features: Age, Blood Pressure, Cholesterol, etc.
Output: Heart Disease Risk (0 = No Risk, 1 = Risk)
🧾 PDF Report Includes
Patient Name
Contact Number
Prediction Result
Risk Level
Model Explanation Summary


Built with ❤️ by Ayushi
