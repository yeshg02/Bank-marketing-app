import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="💳",
    layout="wide"
)

# ========================
# Load Model & Features
# ========================
model = joblib.load("best_model.pkl")
features = joblib.load("model_features.pkl")

# ========================
# Custom CSS for Styling
# ========================
st.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 22px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 12px;
        background: linear-gradient(90deg, #007bff, #00c6ff);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-card {
        padding: 22px;
        border-radius: 14px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .success-card {
        background-color: #d4edda;
        color: #155724;
        border-left: 8px solid #28a745;
    }
    .error-card {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 8px solid #dc3545;
    }
    .metric-box {
        border-radius: 12px;
        padding: 18px;
        background: linear-gradient(120deg, #00c6ff, #007bff);
        color: white;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========================
# Sidebar Branding
# ========================
st.sidebar.markdown(
    "<div class='sidebar-title'>💳 Bank Marketing App</div>",
    unsafe_allow_html=True
)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135783.png", use_container_width=True)
st.sidebar.markdown("---")

# ========================
# Sidebar Inputs
# ========================
st.sidebar.header("📝 Client Information")

job_options = [
    "admin.", "blue-collar", "entrepreneur", "housemaid",
    "management", "retired", "self-employed", "services",
    "student", "technician", "unemployed", "unknown"
]
marital_options = ["married", "single", "divorced", "unknown"]
education_options = [
    "basic.4y", "basic.6y", "basic.9y", "high.school",
    "illiterate", "professional.course", "university.degree", "unknown"
]
default_options = ["yes", "no", "unknown"]
housing_options = ["yes", "no", "unknown"]
loan_options = ["yes", "no", "unknown"]
contact_options = ["cellular", "telephone"]
month_options = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
poutcome_options = ["failure", "nonexistent", "success"]

user_inputs = {}

# Personal Info
st.sidebar.subheader("👤 Personal Information")
if "age" in features:
    user_inputs["age"] = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
if "job" in features:
    user_inputs["job"] = st.sidebar.selectbox("Job", job_options)
if "marital" in features:
    user_inputs["marital"] = st.sidebar.selectbox("Marital Status", marital_options)
if "education" in features:
    user_inputs["education"] = st.sidebar.selectbox("Education", education_options)

# Financial Info
st.sidebar.subheader("🏦 Financial Information")
if "default" in features:
    user_inputs["default"] = st.sidebar.radio("Credit in Default?", default_options)
if "housing" in features:
    user_inputs["housing"] = st.sidebar.radio("Housing Loan?", housing_options)
if "loan" in features:
    user_inputs["loan"] = st.sidebar.radio("Personal Loan?", loan_options)

# Contact & Campaign Info
st.sidebar.subheader("📞 Contact & Campaign Info")
if "contact" in features:
    user_inputs["contact"] = st.sidebar.selectbox("Contact Communication", contact_options)
if "month" in features:
    user_inputs["month"] = st.sidebar.selectbox("Last Contact Month", month_options)
if "poutcome" in features:
    user_inputs["poutcome"] = st.sidebar.selectbox("Previous Campaign Outcome", poutcome_options)

# Any remaining features (numeric)
for feature in features:
    if feature not in user_inputs:
        user_inputs[feature] = st.sidebar.number_input(f"{feature}", value=0)

input_df = pd.DataFrame([user_inputs])

# ========================
# Main Layout - Tabs
# ========================
st.title("💰 Bank Marketing Subscription Prediction Dashboard")
st.markdown(
    "This app helps predict whether a bank client will subscribe to a term deposit. "
    "Navigate between **Prediction** and **Insights** using the tabs below."
)

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Insights"])

# --- Prediction Tab
with tab1:
    st.subheader("Make a Prediction")

    if st.button("Run Prediction"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Prediction Result")
            if prediction == 1:
                st.markdown(
                    f"<div class='prediction-card success-card'>✅ The client is <b>likely to subscribe</b><br> Probability: {probability:.2f}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='prediction-card error-card'>❌ The client is <b>not likely to subscribe</b><br> Probability: {probability:.2f}</div>",
                    unsafe_allow_html=True
                )

        with col2:
            st.subheader("Model Confidence")
            st.markdown(
                f"<div class='metric-box'>{probability:.2%}</div>",
                unsafe_allow_html=True
            )

# --- Insights Tab
with tab2:
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        feat_importances = pd.Series(model.feature_importances_, index=features)
        top_features = feat_importances.sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(6, 4))
        top_features.plot(kind='barh', ax=ax, color="skyblue", edgecolor="black")
        ax.set_title("Top 10 Important Features", fontsize=14, weight="bold")
        st.pyplot(fig)

# ========================
# Footer
# ========================
st.markdown("---")
st.markdown("🌟 Made with ❤️ using Streamlit | Project by **Yesh Reddy**")
