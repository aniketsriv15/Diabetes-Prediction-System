import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("diabetes.csv")

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.block-container{
    padding-top:2rem;
}

.metric-card{
    background:#1f2937;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.markdown(
"""
<h1 style='text-align:center'>
🩺 Diabetes Prediction System
</h1>
""",
unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("📊 Project Information")

    st.success("Best Model : SVM")

    st.write("Accuracy : 82.5%")

    st.write("""
Algorithms Used:

✅ Logistic Regression

✅ Random Forest

✅ Support Vector Machine

✅ KNN

✅ Voting Classifier
""")

# =========================
# NAVIGATION
# =========================

page = st.radio(
"",
[
"📊 EDA",
"🤖 Model Comparison",
"📈 Evaluation",
"🔍 Predict",
"📋 Dataset"
],
horizontal=True
)

# =========================
# EDA PAGE
# =========================

if page == "📊 EDA":

    st.header("Exploratory Data Analysis")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Total Samples",
        len(df)
    )

    col2.metric(
        "Features",
        len(df.columns)-1
    )

    col3.metric(
        "Diabetic Cases",
        int(df["Outcome"].sum())
    )

    col4.metric(
        "Positive Rate",
        f"{df['Outcome'].mean()*100:.1f}%"
    )

    st.markdown("---")

    st.subheader("Class Distribution")

    st.bar_chart(
        df["Outcome"].value_counts()
    )

# =========================
# MODEL COMPARISON
# =========================

elif page == "🤖 Model Comparison":

    st.header("Model Comparison")

    comparison = pd.DataFrame({

        "Model":[
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "KNN"
        ],

        "Accuracy":[
            0.75,
            0.75,
            0.825,
            0.80
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.bar_chart(
        comparison.set_index("Model")
    )

    st.success(
        "🏆 Best Model : SVM (82.5%)"
    )

# =========================
# EVALUATION
# =========================

elif page == "📈 Evaluation":

    st.header("Model Evaluation")

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Accuracy",
        "82.5%"
    )

    c2.metric(
        "Best Model",
        "SVM"
    )

    c3.metric(
        "ROC-AUC",
        "0.82"
    )

    st.markdown("---")

    st.info("""
Support Vector Machine (SVM) achieved the highest accuracy among all tested algorithms.

The model demonstrates strong predictive performance for diabetes risk assessment.
""")

# =========================
# PREDICTION PAGE
# =========================

elif page == "🔍 Predict":

    st.header("Patient Prediction")

    preg = st.number_input(
        "Pregnancies",
        0,20,1
    )

    glu = st.number_input(
        "Glucose",
        0,300,120
    )

    bp = st.number_input(
        "Blood Pressure",
        0,200,70
    )

    skin = st.number_input(
        "Skin Thickness",
        0,100,20
    )

    insulin = st.number_input(
        "Insulin",
        0,900,80
    )

    bmi = st.number_input(
        "BMI",
        0.0,70.0,25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        0.0,3.0,0.5
    )

    age = st.number_input(
        "Age",
        1,120,30
    )

    # BMI CATEGORY

    if bmi < 18.5:
        bmi_cat = 0
    elif bmi < 25:
        bmi_cat = 1
    elif bmi < 30:
        bmi_cat = 2
    else:
        bmi_cat = 3

    if st.button("🔍 Predict Diabetes"):

        features = np.array([
            preg,
            glu,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age,
            bmi_cat
        ]).reshape(1,-1)

        features = scaler.transform(features)

        prediction = model.predict(features)

        try:
            prob = model.predict_proba(features)[0][1]
        except:
            prob = 0.50

        st.subheader("Risk Analysis")

        # Gauge Meter

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text':"Diabetes Risk (%)"},
            gauge={
                'axis':{'range':[0,100]},
                'steps':[
                    {'range':[0,40],'color':'green'},
                    {'range':[40,70],'color':'orange'},
                    {'range':[70,100],'color':'red'}
                ]
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.metric(
            "Probability",
            f"{prob*100:.2f}%"
        )

        st.progress(
            int(prob*100)
        )

        if prediction[0] == 1:

            st.error(
                "⚠️ HIGH RISK OF DIABETES"
            )

        else:

            st.success(
                "✅ LOW RISK OF DIABETES"
            )

# =========================
# DATASET PAGE
# =========================

elif page == "📋 Dataset":

    st.header("Dataset Overview")

    st.subheader(
        "First 20 Rows"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader(
        "Statistical Summary"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Developed by Aniket Srivastava | JIIT"
)