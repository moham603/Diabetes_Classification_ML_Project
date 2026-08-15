import streamlit as st
import pandas as pd
import joblib


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# Custom CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
   GENERAL
===================================================== */

.stApp {
    background: #F5F8FC;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =====================================================
   MAIN HEADER
===================================================== */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #17324D;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    font-weight: 500;
    color: #64748B;
    margin-bottom: 35px;
}


/* =====================================================
   SECTION HEADER
===================================================== */

.section-header {
    background: white;
    padding: 16px 20px;
    border-radius: 14px;
    margin-top: 15px;
    margin-bottom: 18px;
    border-left: 5px solid #2483E2;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
}

.section-title {
    font-size: 22px;
    font-weight: 750;
    color: #17324D;
    margin: 0;
}


/* =====================================================
   LABELS
===================================================== */

.stNumberInput label,
.stSelectbox label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #263B53 !important;
}


/* =====================================================
   INPUTS
===================================================== */

.stNumberInput input {
    height: 48px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}

.stSelectbox div[data-baseweb="select"] {
    min-height: 48px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
}


/* =====================================================
   INPUT FOCUS
===================================================== */

.stNumberInput input:focus {
    border-color: #2483E2 !important;
    box-shadow: 0 0 0 2px rgba(36, 131, 226, 0.15) !important;
}


/* =====================================================
   PREDICTION BUTTON
===================================================== */

.stButton {
    width: 100%;
    margin-top: 25px;
}

.stButton > button {
    width: 100% !important;
    height: 58px !important;
    border-radius: 14px !important;
    border: none !important;

    background: #2483E2 !important;
    color: white !important;

    font-size: 19px !important;
    font-weight: 800 !important;

    box-shadow: 0 6px 18px rgba(36, 131, 226, 0.25);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #1F2EA8 !important;
    color: white !important;

    transform: translateY(-1px);

    box-shadow: 0 8px 22px rgba(31, 46, 168, 0.25);
}


/* =====================================================
   RESULT CARD
===================================================== */

.result-card {
    margin-top: 30px;
    padding: 32px;
    border-radius: 18px;
    text-align: center;
    background: white;
    border: 1px solid #E2E8F0;

    box-shadow:
        0 8px 25px rgba(15, 23, 42, 0.08);
}

.result-label {
    font-size: 18px;
    font-weight: 700;
    color: #64748B;
    margin-bottom: 10px;
}

.result-value {
    font-size: 38px;
    font-weight: 850;
}

.diabetic-result {
    color: #DC2626;
}

.non-diabetic-result {
    color: #059669;
}


/* =====================================================
   ALERTS
===================================================== */

.stAlert {
    margin-top: 15px;
    border-radius: 12px;
    font-size: 15px;
    font-weight: 600;
}


/* =====================================================
   INFORMATION BOX
===================================================== */

.info-box {
    margin-top: 30px;
    padding: 17px 20px;
    border-radius: 12px;

    background: #EAF4FF;
    border: 1px solid #C9E2FF;

    color: #334155;

    font-size: 14px;
    line-height: 1.6;
}


/* =====================================================
   FOOTER
===================================================== */

.footer {
    text-align: center;
    margin-top: 25px;
    color: #94A3B8;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL - CACHED
# =========================================================

@st.cache_resource
def load_model():
    """
    Load the trained ML model and preprocessing objects
    only once and keep them cached.
    """

    model_data = joblib.load("diabetes_model.pkl")

    model = model_data["model"]
    scaler = model_data["scaler"]
    gender_encoder = model_data["gender_encoder"]
    class_encoder = model_data["class_encoder"]

    return (
        model,
        scaler,
        gender_encoder,
        class_encoder
    )


# Load model
model, scaler, gender_encoder, class_encoder = load_model()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🩺 Diabetes Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Diabetes Classification'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INFORMATION
# =========================================================

st.markdown(
    """
    <div class="section-header">
        <div class="section-title">
            👤 Patient Information
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ROW 1
# =========================================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=40,
        step=1
    )


with col2:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


# =========================================================
# ROW 2
# =========================================================

col1, col2 = st.columns(2)

with col1:

    blood_sugar = st.number_input(
        "🩸 Blood Sugar Level",
        min_value=0.0,
        max_value=1000.0,
        value=120.0,
        step=1.0
    )


with col2:

    creatinine = st.number_input(
        "🧪 Creatinine",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.1
    )


# =========================================================
# ROW 3
# =========================================================

col1, col2 = st.columns(2)

with col1:

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )


with col2:

    urea = st.number_input(
        "💧 Urea",
        min_value=0.0,
        max_value=300.0,
        value=30.0,
        step=1.0
    )


# =========================================================
# ROW 4
# =========================================================

col1, col2 = st.columns(2)

with col1:

    cholesterol = st.number_input(
        "🧬 Cholesterol",
        min_value=0.0,
        max_value=500.0,
        value=190.0,
        step=1.0
    )


with col2:

    hba1c = st.number_input(
        "📊 HbA1c",
        min_value=0.0,
        max_value=20.0,
        value=5.5,
        step=0.1
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔍  Predict Diabetes",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # Encode Gender
    # -----------------------------------------------------

    gender_encoded = gender_encoder.transform(
        [gender]
    )[0]


    # -----------------------------------------------------
    # Create Input DataFrame
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "Age": [age],

        "Gender": [gender_encoded],

        "Blood_Sugar_Level": [blood_sugar],

        "Creatinine": [creatinine],

        "BMI": [bmi],

        "Urea": [urea],

        "Cholesterol": [cholesterol],

        "HbA1c": [hba1c]

    })


    # -----------------------------------------------------
    # Scale Input
    # -----------------------------------------------------

    input_scaled = scaler.transform(
        input_data
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = model.predict(
        input_scaled
    )


    # -----------------------------------------------------
    # Convert Prediction to Original Class
    # -----------------------------------------------------

    result = class_encoder.inverse_transform(
        prediction
    )[0]


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    if result == "Diabetic":

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">'
            'Prediction Result'
            '</div>'
            '<div class="result-value diabetic-result">'
            '🔴 Diabetic'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.warning(
            "The model predicts that the patient "
            "is likely to be diabetic."
        )


    else:

        st.markdown(
            '<div class="result-card">'
            '<div class="result-label">'
            'Prediction Result'
            '</div>'
            '<div class="result-value non-diabetic-result">'
            '🟢 Non-Diabetic'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        st.success(
            "The model predicts that the patient "
            "is likely to be non-diabetic."
        )


# =========================================================
# INFORMATION / DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="info-box">

        💡 <b>Note:</b>

        This application is a machine learning prediction
        tool for educational purposes and should not be
        considered a medical diagnosis.

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Diabetes Classification Project • SVM Model
    </div>
    """,
    unsafe_allow_html=True
)