from reportlab.platypus import SimpleDocTemplate
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from openai import OpenAI
import os
import re

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Heart AI", page_icon="💓", layout="wide")

client = OpenAI(api_key=os.getenv("key"))
# ======================
# LOAD MODELS
# ======================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
explainer = pickle.load(open("explainer.pkl", "rb"))

try:
    feature_names = pickle.load(open("features.pkl", "rb"))
except:
    feature_names = [
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal"
    ]

# ======================
# SESSION STATE
# ======================
for key in ["result", "explanation", "pdf_file", "scaled", "advice_done", "advice_text"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ======================
# UI
# ======================
st.markdown("<h1 style='text-align:center;'>💓 Heart AI Predictor</h1>", unsafe_allow_html=True)

# ======================
# INPUT
# ======================
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Full Name")

    contact = st.text_input("Contact Number (10 digits only)")

    age = st.number_input("Age", 1, 120, 25)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.slider("Chest Pain", 0, 3, 0)
    trestbps = st.number_input("BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 180)
    fbs = st.selectbox("FBS", ["No", "Yes"])

with col2:
    restecg = st.slider("Rest ECG", 0, 2, 0)
    thalach = st.number_input("Max HR", 60, 220, 150)
    exang = st.selectbox("Angina", ["No", "Yes"])
    oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.slider("Slope", 0, 2, 1)
    ca = st.slider("CA", 0, 3, 0)
    thal = st.slider("Thal", 0, 3, 2)

# ======================
# VALIDATION
# ======================

# Name validation (full name)
if name and len(name.split()) < 2:
    st.error("⚠️ Please enter full name (First + Last)")
    st.stop()

# Contact validation (10 digits only)
if contact and not re.fullmatch(r"\d{10}", contact):
    st.error("⚠️ Contact must be exactly 10 digits")
    st.stop()

# ======================
# ENCODING
# ======================
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

# ======================
# PDF FUNCTION
# ======================
def generate_pdf(data, result, explanation):
    pdf = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Heart Disease Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Name: {data.get('name','')}", styles["Normal"]))
    content.append(Paragraph(f"Contact: {data.get('contact','')}", styles["Normal"]))
    content.append(Spacer(1, 12))

    for k, v in data.items():
        if k not in ["name", "contact"]:
            content.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Prediction: {result}", styles["Heading2"]))
    content.append(Paragraph(explanation, styles["Normal"]))

    pdf.build(content)
    return "report.pdf"

# ======================
# AI ADVICE
# ======================
def get_advice(data, result):
    try:
        prompt = f"""
        Patient Data: {data}
        Prediction: {result}

        Give:
        1. Simple reason
        2. 2 precautions
        3. Keep it short
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )

        return response.choices[0].message.content

    except:
        return "AI temporarily unavailable"

# ======================
# PREDICT BUTTON
# ======================
if st.button("🚀 Predict"):

    input_data = [[
        age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak,
        slope, ca, thal
    ]]

    df = pd.DataFrame(input_data, columns=feature_names)
    scaled = scaler.transform(df.astype(float))

    st.session_state.scaled = scaled
    st.session_state.df = df

    pred = model.predict(scaled)
    proba = model.predict_proba(scaled)[0]

    if pred[0] == 0:
        result = f"Low Risk ({proba[0]*100:.2f}%)"
        st.success(result)
    else:
        result = f"High Risk ({proba[1]*100:.2f}%)"
        st.error(result)

    st.session_state.result = result

    # ======================
    # SHAP
    # ======================
    shap_values = explainer(scaled)
    sv = shap_values[0]

    if len(sv.values.shape) == 2:
        sv = sv[:, 1]

    vals = np.array(sv.values).reshape(-1)
    names = feature_names[:len(vals)]

    top = sorted(
        zip(names, vals),
        key=lambda x: abs(float(x[1])),
        reverse=True
    )[:3]

    explanation = "Top factors: " + ", ".join(
        [f"{f}({round(float(v),2)})" for f, v in top]
    )

    st.session_state.explanation = explanation
    st.write(explanation)

    fig, ax = plt.subplots()
    shap.plots.waterfall(sv, show=False)
    st.pyplot(fig)
    plt.close(fig)

# ======================
# PDF GENERATE
# ======================
if st.button("📄 Generate PDF"):

    if st.session_state.result is None:
        st.warning("Run prediction first")

    else:
        pdf_file = generate_pdf(
            {
                "name": name,
                "contact": contact,
                "age": age,
                "sex": sex,
                "cp": cp,
                "bp": trestbps,
                "chol": chol
            },
            st.session_state.result,
            st.session_state.explanation
        )

        st.session_state.pdf_file = pdf_file
        st.success("PDF created!")

# ======================
# DOWNLOAD
# ======================
if st.session_state.pdf_file:
    with open(st.session_state.pdf_file, "rb") as f:
        st.download_button(
            "📥 Download Report",
            f,
            file_name="report.pdf"
        )

# ======================
# AI ADVICE BUTTON (GREEN STATE)
# ======================
if st.session_state.result:
    if st.button("🩺 Get Advice"):
        st.session_state.advice_text = get_advice(
            {"name": name, "age": age, "cp": cp},
            st.session_state.result
        )
        st.session_state.advice_done = True

# ======================
# ADVICE OUTPUT UI
# ======================
if st.session_state.advice_done:
    st.success("🟢 ADVISED")

    if "High Risk" in st.session_state.result:
        st.error("⚠️ Consult to doctor immediately")
    else:
        st.success("💚 You are fine. Take care")

    st.info(st.session_state.advice_text)