import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1040 60%, #0f0c29 100%);
    color: #e8e0f0;
}

/* ── Hide default streamlit bits ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(20, 10, 50, 0.95) !important;
    border-right: 1px solid rgba(167,139,250,0.2);
}
[data-testid="stSidebar"] * { color: #c4b5fd !important; }

/* ── Custom header banner ── */
.header-banner {
    background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(79,70,229,0.2));
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    backdrop-filter: blur(10px);
}
.header-icon {
    font-size: 52px;
    line-height: 1;
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, #c4b5fd, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.header-sub {
    font-size: 14px;
    color: #8b7ab8;
    margin: 6px 0 0 0;
}

/* ── Section cards ── */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 15px;
    font-weight: 600;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 18px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(167,139,250,0.15);
}

/* ── Input labels ── */
label, .stNumberInput label, div[data-testid="stFormLabel"] {
    color: #c4b5fd !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}

/* ── Number inputs ── */
input[type="number"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.25) !important;
    border-radius: 8px !important;
    color: #e8e0f0 !important;
    font-family: 'Inter', sans-serif !important;
}
input[type="number"]:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(167,139,250,0.25) !important;
    border-radius: 8px !important;
    color: #e8e0f0 !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 40px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
}

/* ── Success / error boxes ── */
.stSuccess {
    background: rgba(34,197,94,0.1) !important;
    border: 1px solid rgba(34,197,94,0.3) !important;
    border-radius: 12px !important;
    color: #86efac !important;
}
.stError {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 12px !important;
}

/* ── Result display ── */
.result-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(79,70,229,0.1));
    border: 1px solid rgba(167,139,250,0.35);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    margin-top: 16px;
    box-shadow: 0 0 40px rgba(124,58,237,0.15);
}
.result-label {
    font-size: 13px;
    color: #8b7ab8;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}
.result-value {
    font-family: 'Playfair Display', serif;
    font-size: 80px;
    font-weight: 700;
    background: linear-gradient(135deg, #c4b5fd, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 10px;
}
.result-value-grade {
    font-family: 'Playfair Display', serif;
    font-size: 96px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 10px;
}
.result-caption {
    font-size: 15px;
    color: #a78bfa;
}

/* ── Info badges ── */
.info-badge {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #a78bfa;
    margin: 4px;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(167,139,250,0.12);
    margin: 24px 0;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 20px;
}
.metric-card {
    flex: 1;
    min-width: 120px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.metric-val {
    font-size: 22px;
    font-weight: 700;
    color: #c4b5fd;
    font-family: 'Playfair Display', serif;
}
.metric-lbl {
    font-size: 11px;
    color: #6b5a8a;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    knn = joblib.load("knn_regression.pkl")
    gnb = joblib.load("gaussian_nb.pkl")
    scaler = joblib.load("scaler.pkl")
    return knn, gnb, scaler


try:
    knn_model, gnb_model, scaler = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose Algorithm",
        ["KNN Regression", "GaussianNB Classification"],
        help="KNN → Predicts a numeric score | GaussianNB → Predicts grade class"
    )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    if model_choice == "KNN Regression":
        st.info(
            "**KNN Regression**\nPredicts a continuous performance score based on nearest neighbors.")
    else:
        st.info(
            "**GaussianNB Classifier**\nPredicts a grade class (0–4) using Gaussian Naive Bayes.")

    st.markdown("---")
    st.markdown("### 📋 Feature Guide")
    guide = {
        "Gender": "0 = Female, 1 = Male",
        "Ethnicity": "0 to 3",
        "ParentalEducation": "0=None → 4=Postgrad",
        "Tutoring": "0=No, 1=Yes",
        "ParentalSupport": "0=None → 4=Very High",
        "Extracurricular": "0=No, 1=Yes",
        "Sports / Music / Volunteering": "0=No, 1=Yes",
        "GPA": "0.0 to 4.0",
    }
    for k, v in guide.items():
        st.markdown(
            f"<span class='info-badge'>**{k}:** {v}</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div class="header-icon">🎓</div>
    <div>
        <p class="header-title">Student Performance Predictor</p>
        <p class="header-sub">ML-powered academic outcome analysis · KNN Regression & GaussianNB Classification</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Model not loaded warning
if not models_loaded:
    st.error(
        f"⚠️ Model files not found! Make sure `knn_regression.pkl`, `gaussian_nb.pkl`, and `scaler.pkl` are in the same folder.\n\n**Error:** {load_error}")
    st.stop()


# ─────────────────────────────────────────────
#  INPUT SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-card"><p class="section-title">🧾 Student Feature Inputs</p>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Info**")
    age = st.number_input("🎂 Age",               min_value=10,
                          max_value=25,  value=16,  step=1)
    gender = st.number_input(
        "👤 Gender",            min_value=0,   max_value=1,   value=0,   step=1)
    ethnicity = st.number_input(
        "🌍 Ethnicity",         min_value=0,   max_value=3,   value=0,   step=1)
    parental_edu = st.number_input(
        "🎓 ParentalEducation", min_value=0,   max_value=4,   value=2,   step=1)
    parental_sup = st.number_input(
        "👨‍👩‍👧 ParentalSupport",  min_value=0,   max_value=4,   value=2,   step=1)

with col2:
    st.markdown("**📚 Academic Info**")
    study_time = st.number_input(
        "📚 StudyTimeWeekly",   min_value=0.0, max_value=40.0, value=10.0, step=0.5, format="%.1f")
    absences = st.number_input(
        "🏫 Absences",          min_value=0,   max_value=50,   value=3,   step=1)
    tutoring = st.number_input(
        "🧑‍🏫 Tutoring",         min_value=0,   max_value=1,    value=0,   step=1)
    gpa = st.number_input("📊 GPA",               min_value=0.0,
                          max_value=4.0,  value=2.5, step=0.01, format="%.2f")

with col3:
    st.markdown("**🎭 Activities**")
    extracurricular = st.number_input(
        "🎭 Extracurricular",   min_value=0,   max_value=1,   value=0,   step=1)
    sports = st.number_input(
        "⚽ Sports",            min_value=0,   max_value=1,   value=0,   step=1)
    music = st.number_input(
        "🎵 Music",             min_value=0,   max_value=1,   value=0,   step=1)
    volunteering = st.number_input(
        "🤝 Volunteering",      min_value=0,   max_value=1,   value=0,   step=1)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LIVE SUMMARY
# ─────────────────────────────────────────────
activity_count = extracurricular + sports + music + volunteering
support_label = ["None", "Low", "Moderate",
                 "High", "Very High"][int(parental_sup)]

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card"><div class="metric-val">{age}</div><div class="metric-lbl">Age</div></div>
  <div class="metric-card"><div class="metric-val">{gpa:.2f}</div><div class="metric-lbl">GPA</div></div>
  <div class="metric-card"><div class="metric-val">{study_time:.0f}h</div><div class="metric-lbl">Study / Week</div></div>
  <div class="metric-card"><div class="metric-val">{absences}</div><div class="metric-lbl">Absences</div></div>
  <div class="metric-card"><div class="metric-val">{activity_count}</div><div class="metric-lbl">Activities</div></div>
  <div class="metric-card"><div class="metric-val">{support_label}</div><div class="metric-lbl">Parent Support</div></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREDICT BUTTON
# ─────────────────────────────────────────────
st.markdown("")
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button("🔍 Predict Performance")


# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
if predict_btn:
    columns = [
        "Age", "Gender", "Ethnicity", "ParentalEducation",
        "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
        "Extracurricular", "Sports", "Music", "Volunteering", "GPA"
    ]
    feature_values = [
        age, gender, ethnicity, parental_edu,
        study_time, absences, tutoring, parental_sup,
        extracurricular, sports, music, volunteering, gpa
    ]

    input_df = pd.DataFrame([feature_values], columns=columns)

    try:
        input_scaled = scaler.transform(input_df)

        if model_choice == "KNN Regression":
            prediction = knn_model.predict(input_scaled)[0]

            if prediction >= 80:
                emoji, msg, color = "🌟", "Excellent Performance!", "#22c55e"
            elif prediction >= 65:
                emoji, msg, color = "👍", "Good Performance", "#3b82f6"
            elif prediction >= 50:
                emoji, msg, color = "📈", "Average Performance", "#eab308"
            else:
                emoji, msg, color = "⚠️", "Needs Improvement", "#ef4444"

            bar_pct = min(100, max(0, prediction))

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">KNN Regression · Predicted Score</div>
                <div class="result-value">{prediction:.2f}</div>
                <div class="result-caption">{emoji} {msg}</div>
                <div style="height:10px; border-radius:5px; background:rgba(255,255,255,0.06);
                            max-width:400px; margin:20px auto 0 auto; overflow:hidden;">
                    <div style="height:100%; width:{bar_pct}%; border-radius:5px;
                                background:linear-gradient(to right, #7c3aed, {color});
                                transition: width 1s ease;"></div>
                </div>
                <div style="font-size:12px; color:#6b5a8a; margin-top:8px;">Score out of 100</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            prediction = gnb_model.predict(input_scaled)[0]
            grade_map = {0: ("F", "#ef4444"), 1: ("D", "#f97316"),
                         2: ("C", "#eab308"), 3: ("B", "#22c55e"), 4: ("A", "#3b82f6")}
            grade_label, grade_color = grade_map.get(
                int(prediction), ("?", "#a78bfa"))

            grade_badges = ""
            for cls, (lbl, clr) in grade_map.items():
                if cls == int(prediction):
                    grade_badges += f'<span style="display:inline-block;width:44px;height:44px;line-height:44px;border-radius:10px;text-align:center;font-size:20px;font-weight:700;background:{clr}30;border:2px solid {clr};color:{clr};margin:4px;transform:scale(1.2)">{lbl}</span>'
                else:
                    grade_badges += f'<span style="display:inline-block;width:44px;height:44px;line-height:44px;border-radius:10px;text-align:center;font-size:18px;font-weight:700;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);color:#4a3a6a;margin:4px">{lbl}</span>'

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">GaussianNB Classification · Grade Class {int(prediction)}</div>
                <div class="result-value-grade" style="color:{grade_color};
                     text-shadow:0 0 40px {grade_color}60;">{grade_label}</div>
                <div class="result-caption" style="margin-bottom:20px;">Predicted Grade — Class {int(prediction)}</div>
                <div>{grade_badges}</div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.info("💡 Make sure all .pkl files are in the same directory as app.py")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; font-size:12px; color:#4a3a6a; font-family:'Inter',sans-serif; line-height:2;">
    <div>🎓 Student Performance Prediction · ML Project</div>
    <div>Models: <code style="color:#7c3aed">knn_regression.pkl</code> &nbsp;|&nbsp;
         <code style="color:#7c3aed">gaussian_nb.pkl</code> &nbsp;|&nbsp;
         <code style="color:#7c3aed">scaler.pkl</code></div>
</div>
""", unsafe_allow_html=True)
