import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ─────────────────────────────────────────
#  Page Config  (must be first st call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  Custom CSS  – dark academic aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0e1117;
    --surface:   #161b27;
    --border:    #252d3d;
    --accent:    #5b8dee;
    --accent2:   #e8a838;
    --text:      #dde3f0;
    --muted:     #7a87a0;
    --success:   #3ecf8e;
    --danger:    #f56565;
    --radius:    12px;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Selectbox & Number inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #1e2535 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
}

/* ── Buttons ── */
div.stButton > button {
    width: 100%;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.02em;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.85; }

/* ── Section card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a2340 0%, #0e1117 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    padding: 2rem 2.4rem;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem !important;
    color: #fff !important;
    margin: 0 0 0.3rem 0;
}
.hero p { color: var(--muted); margin: 0; font-size: 0.95rem; }

/* ── Label style ── */
.field-label {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.2rem;
}

/* ── Result box ── */
.result-box {
    border-radius: var(--radius);
    padding: 1.4rem 2rem;
    font-size: 1.3rem;
    font-weight: 600;
    text-align: center;
    margin-top: 1rem;
}
.result-success {
    background: rgba(62, 207, 142, 0.12);
    border: 1px solid var(--success);
    color: var(--success);
}
.result-class {
    background: rgba(91, 141, 238, 0.12);
    border: 1px solid var(--accent);
    color: var(--accent);
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Tooltip badge ── */
.badge {
    display: inline-block;
    background: rgba(91,141,238,0.15);
    color: var(--accent);
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin-left: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Load Models
# ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    knn = joblib.load("knn_regression.pkl")
    gnb = joblib.load("gaussian_nb.pkl")
    scaler = joblib.load("scaler.pkl")
    return knn, gnb, scaler


try:
    knn_model, gnb_model, scaler = load_models()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ─────────────────────────────────────────
#  Feature Definitions
# ─────────────────────────────────────────
FEATURES = {
    "Age":               {"min": 10, "max": 25, "default": 17, "step": 1,    "tip": "Student age in years"},
    "Gender":            {"min": 0,  "max": 1,  "default": 0,  "step": 1,    "tip": "0 = Male, 1 = Female"},
    "Ethnicity":         {"min": 0,  "max": 3,  "default": 0,  "step": 1,    "tip": "Encoded ethnicity group (0–3)"},
    "ParentalEducation": {"min": 0,  "max": 4,  "default": 2,  "step": 1,    "tip": "0=None → 4=Higher Education"},
    "StudyTimeWeekly":   {"min": 0.0, "max": 40.0, "default": 10.0, "step": 0.5,  "tip": "Hours studied per week"},
    "Absences":          {"min": 0,  "max": 60, "default": 3,  "step": 1,    "tip": "Number of school absences"},
    "Tutoring":          {"min": 0,  "max": 1,  "default": 0,  "step": 1,    "tip": "0 = No, 1 = Yes"},
    "ParentalSupport":   {"min": 0,  "max": 4,  "default": 2,  "step": 1,    "tip": "0=None → 4=Very High"},
    "Extracurricular":   {"min": 0,  "max": 1,  "default": 0,  "step": 1,    "tip": "0 = No, 1 = Yes"},
    "Sports":            {"min": 0,  "max": 1,  "default": 0,  "step": 1,    "tip": "0 = No, 1 = Yes"},
    "Music":             {"min": 0,  "max": 1,  "default": 0,  "step": 1,    "tip": "0 = No, 1 = Yes"},
    "Volunteering":      {"min": 0,  "max": 1,  "default": 0,  "step": 1,    "tip": "0 = No, 1 = Yes"},
    "GPA":               {"min": 0.0, "max": 4.0, "default": 2.5, "step": 0.01,  "tip": "Grade Point Average (0.0 – 4.0)"},
}

COLUMNS = list(FEATURES.keys())


# ─────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.6rem'>
        <div style='font-size:2.8rem'>🎓</div>
        <div style='font-family: Playfair Display, serif; font-size:1.2rem; font-weight:700; color:#dde3f0'>
            Student Predictor
        </div>
        <div style='color:#7a87a0; font-size:0.8rem; margin-top:0.3rem'>ML-Powered Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Select Model",
        ["KNN Regression", "GaussianNB Classification"],
        help="KNN predicts a continuous GradeClass score; GaussianNB predicts a category."
    )

    st.markdown("---")

    st.markdown("""
    <div style='color:#7a87a0; font-size:0.8rem; line-height:1.6'>
    <b style='color:#dde3f0'>KNN Regression</b><br>
    Predicts a numeric performance score based on K nearest neighbours.<br><br>
    <b style='color:#dde3f0'>GaussianNB Classification</b><br>
    Classifies the student into a performance grade category.
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Model load failed:\n{load_error}")


# ─────────────────────────────────────────
#  Main — Hero
# ─────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>📊 Student Performance Predictor</h1>
    <p>Fill in the student profile below and click <strong>Predict</strong> to get an instant ML-based performance estimate.</p>
</div>
""", unsafe_allow_html=True)

# Status indicator
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.metric("Model", model_choice.split()[0], delta=None)
with col_s2:
    status = "✅ Ready" if model_loaded else "❌ Not Loaded"
    st.metric("Status", status)
with col_s3:
    st.metric("Features", len(COLUMNS))

st.markdown("---")


# ─────────────────────────────────────────
#  Input Form — 3-column grid
# ─────────────────────────────────────────
st.markdown("### 🧾 Student Profile")

features = []
cols_per_row = 3
keys = COLUMNS

for row_start in range(0, len(keys), cols_per_row):
    row_keys = keys[row_start: row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, key in zip(cols, row_keys):
        f = FEATURES[key]
        with col:
            val = st.number_input(
                label=f"**{key}**",
                min_value=float(f["min"]),
                max_value=float(f["max"]),
                value=float(f["default"]),
                step=float(f["step"]),
                help=f["tip"],
                key=key,
            )
            features.append(val)

st.markdown("")

# ─────────────────────────────────────────
#  Predict Button + Result
# ─────────────────────────────────────────
col_btn, col_space = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button(
        "🔍 Predict Performance", use_container_width=True)

if predict_clicked:
    if not model_loaded:
        st.error("❌ Models are not loaded. Please check your .pkl files.")
    else:
        input_df = pd.DataFrame([features], columns=COLUMNS)

        with st.spinner("Running prediction..."):
            try:
                input_scaled = scaler.transform(input_df)

                if model_choice == "KNN Regression":
                    prediction = knn_model.predict(input_scaled)
                    val = prediction[0]
                    st.markdown(f"""
                    <div class='result-box result-success'>
                        📈 Predicted Score &nbsp;→&nbsp; <span style='font-size:1.6rem'>{val:.4f}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Mini bar gauge
                    pct = min(max(val / 4.0, 0), 1)  # assuming 0–4 scale
                    bar_color = "#3ecf8e" if pct >= 0.5 else "#e8a838"
                    st.markdown(f"""
                    <div style='margin-top:1rem'>
                        <div style='font-size:0.78rem; color:#7a87a0; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.4rem'>
                            Performance Gauge (0 – 4)
                        </div>
                        <div style='background:#1e2535; border-radius:8px; height:12px; overflow:hidden'>
                            <div style='width:{pct*100:.1f}%; background:{bar_color}; height:100%; border-radius:8px; transition:width 0.4s ease'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    prediction = gnb_model.predict(input_scaled)
                    label = prediction[0]
                    # Map numeric class to readable label if needed
                    grade_map = {0: "A – Excellent 🏆", 1: "B – Good 👍", 2: "C – Average 📚",
                                 3: "D – Below Average ⚠️", 4: "F – Fail ❌"}
                    readable = grade_map.get(int(label), str(label)) if isinstance(
                        label, (int, float, np.integer)) else str(label)
                    st.markdown(f"""
                    <div class='result-box result-class'>
                        🎯 Predicted Grade Class &nbsp;→&nbsp; {readable}
                    </div>
                    """, unsafe_allow_html=True)

                # Show input summary
                with st.expander("📋 View Input Summary", expanded=False):
                    summary_df = pd.DataFrame({
                        "Feature": COLUMNS,
                        "Value":   features,
                        "Description": [FEATURES[k]["tip"] for k in COLUMNS]
                    })
                    st.dataframe(
                        summary_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")

# ─────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#7a87a0; font-size:0.8rem; padding:0.5rem'>
    Student Performance Predictor &nbsp;·&nbsp; KNN & GaussianNB &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
