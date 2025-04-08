import streamlit as st
import joblib
import numpy as np
from catboost import CatBoostClassifier, Pool
from recommendation import get_recommendations_by_param

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'severity_result' not in st.session_state:
    st.session_state.severity_result = ''
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = ''
if 'show_ranges' not in st.session_state:
    st.session_state.show_ranges = False

# Load models
@st.cache_resource
def load_models():
    try:
        catboost_model = CatBoostClassifier()
        catboost_model.load_model("./pcos_catboost_model.cbm")
    except Exception as e:
        st.error(f"❌ Error loading PCOS model: {e}")
        catboost_model = None

    try:
        severity_model = joblib.load("severity_classifier_model.pkl")
    except Exception as e:
        st.error(f"❌ Error loading severity model: {e}")
        severity_model = None

    return catboost_model, severity_model

catboost_model, severity_model = load_models()

# UI Styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #a3123b;
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #000000 !important;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
select, input, textarea {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 2px solid #FFFFFF !important;
    border-radius: 5px;
    padding: 8px;
}
button {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 2px solid #FFFFFF !important;
    border-radius: 8px;
    padding: 10px;
    font-weight: bold;
}
button:hover {
    background-color: #a3123b !important;
}
.result-box {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    padding: 25px;
    border-radius: 15px;
    font-size: 22px;
    font-weight: bold;
    box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
    margin-top: 15px;
    max-width: 500px;
}
.toggle-btn {
    position: absolute;
    top: 15px;
    right: 15px;
    background: none;
    border: none;
    font-size: 24px;
    color: white;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Function to check if any input value is zero or invalid
def is_input_zero(inputs):
    for value in inputs[:6]:  # Only checking the first six inputs
        if value == 0:
            return True
    return False

# Main page
if st.session_state.page == 'main':
    st.title("PCOS Detection and Recommendation System")

    if st.button("ℹ️ info", key="info_toggle", help="Show/Hide Normal Ranges"):
        st.session_state.show_ranges = not st.session_state.show_ranges

    if st.session_state.show_ranges:
        st.markdown("<h4 style='color: white;'>🧪 Normal Ranges for Inputs:</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #222; padding: 15px; border-radius: 10px; color: white;">
        - <strong>AMH:</strong> 1.0 – 4.0 ng/mL<br>
        - <strong>I beta-HCG:</strong> &lt; 7.0 mIU/mL<br>
        - <strong>II beta-HCG:</strong> &lt; 7.0 mIU/mL <br>
        - <strong>Age:</strong> 18 – 45 years (typically fertile age range)<br>
        - <strong>BMI:</strong> 18.5 – 24.9 kg/m² (normal weight)<br>
        - <strong>Menstrual Cycle Length:</strong> 21 – 35 days<br><br>
        <b>NOTE: The values may vary from person to person. PCOS is detected based on an overall assessment of all parameters.</b>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("This app predicts the likelihood of PCOS and provides tailored recommendations based on severity.")
    st.image("Screenshot 2025-01-27 180205.png")
    st.sidebar.header("Patient Input")

    # Inputs
    amh_ng_ml = st.sidebar.number_input("AMH (ng/mL)", min_value=0.0, step=0.1)
    beta_hcg_i = st.sidebar.number_input("I beta-HCG (mIU/mL)", min_value=0.0, step=0.1)
    beta_hcg_ii = st.sidebar.number_input("II beta-HCG (mIU/mL)", min_value=0.0, step=0.1)
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
    bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    cycle_length = st.sidebar.number_input("Menstrual Cycle Length", min_value=20, max_value=60, step=1)
    hirsutism = st.sidebar.selectbox("Excessive Hair Growth (Hirsutism)", ["No", "Yes"])
    acne = st.sidebar.selectbox("Acne", ["No", "Yes"])

    # Model input
    real_inputs = [
        amh_ng_ml,
        beta_hcg_i,
        beta_hcg_ii,
        age,
        bmi,
        cycle_length,
        1 if hirsutism == "Yes" else 0,
        1 if acne == "Yes" else 0,
    ]

    severity_placeholder = 0
    input_data = [real_inputs + [severity_placeholder]]

    # Predict button
    if st.sidebar.button("Predict", key="predict_button"):
        if is_input_zero(real_inputs):
            st.warning("⚠️ Please enter valid non-zero values to get prediction.")
        elif catboost_model is None:
            st.error("❌ PCOS model is not loaded.")
        else:
            try:
                pool = Pool(data=input_data, feature_names=[
                    "AMH", "I-beta HCG", "II-beta HCG",
                    "Age", "BMI", "Menstrual cycle length", "Excessive hair growth", "Excessive Acne", "Severity"
                ])
                pcos_prob = catboost_model.predict_proba(pool)[0][1]
                threshold = 0.55  # Custom threshold
                pcos_prediction = int(pcos_prob >= threshold)

                prediction_result = (
                    f"✅ PCOS Detected"
                    if pcos_prediction == 1
                    else f"❌ No PCOS"
                )
                st.session_state.prediction_result = prediction_result

                if pcos_prediction == 1 and severity_model:
                    reduced_input = np.array([[amh_ng_ml, beta_hcg_i, beta_hcg_ii, bmi, cycle_length]])
                    severity_probs = severity_model.predict_proba(reduced_input)[0]
                    severity_pred = np.argmax(severity_probs)
                    severity_confidence = severity_probs[severity_pred]
                    severity_levels = ["Mild", "Moderate", "Severe"]
                    severity_result = f"{severity_levels[severity_pred]} ({severity_confidence * 100:.2f}%)"
                    st.session_state.severity_result = severity_result

                    st.session_state.recommendations = get_recommendations_by_param(
                        amh=amh_ng_ml,
                        beta_hcg_1=beta_hcg_i,
                        beta_hcg_2=beta_hcg_ii,
                        bmi=bmi,
                        age=age,
                        cycle_length=cycle_length,
                        severity=severity_result
                    )
                else:
                    st.session_state.severity_result = ''
                    st.session_state.recommendations = []

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    if st.session_state.prediction_result:
        st.markdown(f"<div class='result-box'>🩺 Prediction Result: {st.session_state.prediction_result}</div>", unsafe_allow_html=True)

    if st.session_state.severity_result:
        st.markdown(f"<div class='result-box'>⚠️ PCOS Severity Level: {st.session_state.severity_result}</div>", unsafe_allow_html=True)

    if st.session_state.severity_result:
        if st.button("View Recommendations", key="recommendations_button"):
            st.session_state.page = 'recommendation'

# Recommendation page
elif st.session_state.page == 'recommendation':
    st.title("💡 Lifestyle Recommendations")

    for rec in st.session_state.recommendations:
        st.markdown(f'<h3 style="color: #e6b800;">{rec}</h3>', unsafe_allow_html=True)

        recommendation_text = st.session_state.recommendations[rec]
        lines = recommendation_text.split('\n')
        formatted_text = ""
        for line in lines:
            if line.startswith("-"):
                formatted_text += f'<p style="font-size: 18px; color: #f0f0f0;">* {line[2:].strip()}</p>'
            elif line.startswith("o"):
                formatted_text += f'<p style="font-size: 18px; color: #f0f0f0; margin-left: 20px;">    * {line[2:].strip()}</p>'
            elif line.startswith("**"):
                formatted_text += f'<p style="font-size: 18px; color: #f0f0f0;"><strong>{line[2:].strip()}</strong></p>'
            else:
                formatted_text += f'<p style="font-size: 18px; color: #f0f0f0;">{line.strip()}</p>'

        st.markdown(formatted_text, unsafe_allow_html=True)
        st.markdown("<hr style='border: 1px solid #444444;'>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔙 Back", key="back_button"):
        st.session_state.page = 'main'