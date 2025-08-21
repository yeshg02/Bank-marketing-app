import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# ========================
# Load Model & Preprocessing Objects
# ========================
model = joblib.load("best_model.pkl")
features = joblib.load("model_features.pkl")
encoders = joblib.load("encoders.pkl")   # LabelEncoders for categorical features

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
        padding: 10px;
        background: linear-gradient(90deg, #007bff, #00c6ff);
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .success-card {
        background-color: #d4edda;
        color: #155724;
        border-left: 6px solid #28a745;
    }
    .error-card {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 6px solid #dc3545;
    }
    .metric-box {
        border-radius: 10px;
        padding: 15px;
        background-color: #f1f3f6;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ========================
# Sidebar Branding
# ========================
st.sidebar.markdown("<div class='sidebar-title'>💳 Bank Marketing App</div>", unsafe_allow_html=True)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135783.png", width=120)
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

st.sidebar.subheader("👤 Personal Information")
user_inputs["age"] = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
user_inputs["job"] = st.sidebar.selectbox("Job", job_options)
user_inputs["marital"] = st.sidebar.selectbox("Marital Status", marital_options)
user_inputs["education"] = st.sidebar.selectbox("Education", education_options)

st.sidebar.subheader("🏦 Financial Information")
user_inputs["default"] = st.sidebar.radio("Credit in Default?", default_options)
user_inputs["housing"] = st.sidebar.radio("Housing Loan?", housing_options)
user_inputs["loan"] = st.sidebar.radio("Personal Loan?", loan_options)

st.sidebar.subheader("📞 Contact & Campaign Info")
user_inputs["contact"] = st.sidebar.selectbox("Contact Communication", contact_options)
user_inputs["month"] = st.sidebar.selectbox("Last Contact Month", month_options)
user_inputs["poutcome"] = st.sidebar.selectbox("Previous Campaign Outcome", poutcome_options)

# Add numeric campaign-related inputs
extra_numeric = ["campaign", "pdays", "previous", "emp.var.rate",
                 "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

for feature in extra_numeric:
    user_inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

# Create dataframe
input_df = pd.DataFrame([user_inputs])

# ========================
# Main Layout - Tabs
# ========================
st.title("💰 Bank Marketing Subscription Prediction Dashboard")
st.markdown(
    "This app predicts whether a bank client will subscribe to a term deposit. "
    "Navigate between **Prediction** and **Insights** using the tabs below."
)

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Insights"])

# ========================
# Prediction Tab
# ========================
with tab1:
    st.subheader("Make a Prediction")

    if st.button("Run Prediction"):
        input_encoded = input_df.copy()

        # Apply LabelEncoders with unseen-label handling
        for col, le in encoders.items():
            if col in input_encoded.columns:
                val = input_encoded[col][0]
                if val in le.classes_:
                    input_encoded[col] = le.transform([val])
                else:
                    # Unseen category -> assign -1
                    input_encoded[col] = -1

        # Reindex to match training features
        input_encoded = input_encoded.reindex(columns=features, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        col1, col2 = st.columns([2,1])

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

# ========================
# Insights Tab
# ========================
with tab2:
    st.subheader("Feature Importance")
    if hasattr(model, "feature_importances_"):
        feat_importances = pd.Series(model.feature_importances_, index=features)
        top_features = feat_importances.sort_values(ascending=False).head(10)

        fig, ax = plt.subplots()
        top_features.plot(kind='barh', ax=ax, color="skyblue", edgecolor="black")
        ax.set_title("Top 10 Important Features", fontsize=14, weight="bold")
        st.pyplot(fig)

# ========================
# Footer
# ========================
st.markdown("---")
st.markdown("🌟 Made with ❤️ using Streamlit | Project by **Yesh Reddy**")
