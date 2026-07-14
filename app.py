import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("models/disease_model.pkl")
features = joblib.load("models/features.pkl")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🩺 Disease Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Predict Disease",
        "📊 Analytics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
### Model Information

**Algorithm :** Random Forest Classifier

**Dataset :**
Disease Prediction Dataset (Kaggle)

**Symptoms Used :**
132

**Diseases :**
41
"""
)

st.sidebar.markdown("---")

st.sidebar.success("Developed by\n\n**Ashi Saini**")

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🩺 Disease Prediction System")

    st.write("")

    st.markdown(
    """
    ## Welcome 👋

    This Machine Learning application predicts the **most probable disease**
    based on the symptoms selected by the user.

    The model has been trained using a **Random Forest Classifier**
    on the popular Disease Prediction dataset from Kaggle.

    ---
    """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Features")

        st.markdown("""
        ✅ Predict Disease

        ✅ 132 Symptoms

        ✅ 41 Diseases

        ✅ Random Forest Model

        ✅ Fast Prediction

        ✅ Interactive Dashboard
        """)

    with col2:

        st.subheader("📌 How to Use")

        st.markdown("""
        1. Open **Predict Disease**

        2. Select your symptoms

        3. Click **Predict**

        4. View predicted disease
        """)

    st.markdown("---")

    st.subheader("📖 Diseases Included")

    st.write(
    """
    Fungal Infection, Allergy, GERD, Chronic Cholestasis,
    Drug Reaction, Peptic Ulcer Disease, AIDS, Diabetes,
    Gastroenteritis, Bronchial Asthma, Hypertension,
    Migraine, Cervical Spondylosis, Paralysis,
    Jaundice, Malaria, Chickenpox, Dengue,
    Typhoid, Tuberculosis and many more.
    """
    )

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🔍 Predict Disease":

    st.title("🩺 Disease Prediction")

    st.write(
        "Select the symptoms you are experiencing and click **Predict Disease**."
    )

    st.markdown("---")

    # ===========================================
    # SELECT SYMPTOMS
    # ===========================================

    selected_symptoms = st.multiselect(
        "Select Symptoms",
        options=sorted(features),
        placeholder="Choose one or more symptoms..."
    )

    st.write(f"Selected Symptoms : **{len(selected_symptoms)}**")

    st.markdown("---")

    if st.button("🔍 Predict Disease", use_container_width=True):

        if len(selected_symptoms) == 0:

            st.warning("Please select at least one symptom.")

        else:

            # ===========================================
            # CREATE INPUT VECTOR
            # ===========================================

            input_data = pd.DataFrame(
                [[0] * len(features)],
                columns=features
            )

            for symptom in selected_symptoms:

                input_data.loc[0, symptom] = 1

            # ===========================================
            # PREDICT
            # ===========================================

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]

            confidence = max(probabilities) * 100

            # ===========================================
            # RESULT
            # ===========================================

            st.success("Prediction Completed Successfully!")

            st.markdown("## 🩺 Predicted Disease")

            st.info(f"### {prediction}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(min(confidence / 100, 1.0))

            st.markdown("---")

            # ===========================================
            # TOP 5 PREDICTIONS
            # ===========================================

            st.subheader("Top 5 Possible Diseases")

            probability_df = pd.DataFrame({

                "Disease": model.classes_,

                "Probability (%)": probabilities * 100

            })

            probability_df = probability_df.sort_values(

                by="Probability (%)",

                ascending=False

            ).head(5)

            probability_df["Probability (%)"] = probability_df[
                "Probability (%)"
            ].round(2)

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("---")

            st.subheader("Selected Symptoms")

            st.write(", ".join(selected_symptoms))

# ==========================================================
# ANALYTICS PAGE
# ==========================================================

elif page == "📊 Analytics":

    st.title("📊 Model Analytics")

    st.markdown("### Feature Importance")

    importance = pd.read_csv("models/feature_importance.csv")

    top20 = importance.head(20)

    st.dataframe(
        top20,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("Top 20 Important Symptoms")

    chart_data = top20.set_index("Symptom")

    st.bar_chart(chart_data)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Model",
            value="Random Forest"
        )

        st.metric(
            label="Symptoms",
            value="132"
        )

    with col2:

        st.metric(
            label="Diseases",
            value="41"
        )

        st.metric(
            label="Dataset Size",
            value="4920"
        )

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About")

    st.markdown("""
    ## Disease Prediction System

    This application predicts the most probable disease
    based on symptoms selected by the user.

    The model has been trained using the popular
    Disease Prediction Dataset available on Kaggle.

    ### Technologies Used

    - Python
    - Pandas
    - Scikit-Learn
    - Streamlit
    - Joblib

    ### Machine Learning Algorithm

    - Random Forest Classifier

    ### Dataset

    - 4920 Records
    - 132 Symptoms
    - 41 Diseases

    ### Project Features

    ✔ Disease Prediction

    ✔ Confidence Score

    ✔ Top 5 Predictions

    ✔ Feature Importance

    ✔ Interactive Dashboard
    """)

    st.markdown("---")

    st.subheader("👩‍💻 Developer")

    st.success("""
    **Ashi Saini**

    B.Tech Computer Science Engineering

    Machine Learning Enthusiast
    """)

    st.markdown("---")

    st.info(
        "This project is developed for educational purposes using Machine Learning."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:gray;'>

    Disease Prediction System using Machine Learning ❤️

    Developed by <b>Ashi Saini</b>

    </div>
    """,
    unsafe_allow_html=True
)