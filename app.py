import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🛠️",
    layout="wide"
)

# Load the saved model, scaler and features
@st.cache_resource
def load_artifacts():
    model = joblib.load("rul_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_artifacts()

# Title
st.title("🛠️ AI-Powered Predictive Maintenance System")
st.markdown("### Predict Remaining Useful Life (RUL) of Turbofan Engines using Machine Learning")
st.markdown("---")

# Sidebar
st.sidebar.header("Input Method")
option = st.sidebar.radio("Choose how to input data:", ["Upload CSV File", "Manual Sensor Input"])

# Health status function
def get_health_status(rul):
    if rul > 100:
        return "Healthy", "green", "✅ Engine is in good condition"
    elif rul > 50:
        return "Warning", "orange", "⚠️ Maintenance should be planned soon"
    else:
        return "Critical", "red", "🚨 Immediate maintenance required"

# ===================== OPTION 1: CSV UPLOAD =====================
if option == "Upload CSV File":
    st.subheader("📁 Upload Sensor Data (CSV)")
    st.info("Your CSV file must contain the same sensor columns that were used during training.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of uploaded data:")
            st.dataframe(df.head())

            # Check if required columns exist
            missing_cols = [col for col in features if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns in your CSV: {missing_cols}")
            else:
                # Select only required features
                X = df[features]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)

                result_df = df.copy()
                result_df["Predicted_RUL"] = predictions.round(2)

                # Add health status
                health_list = []
                for rul in predictions:
                    status, _, _ = get_health_status(rul)
                    health_list.append(status)
                result_df["Health_Status"] = health_list

                st.success("✅ Predictions completed successfully!")
                st.write("### Prediction Results:")
                st.dataframe(result_df)

                # Download button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="rul_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ===================== OPTION 2: MANUAL INPUT =====================
else:
    st.subheader("✍️ Manual Sensor Input")
    st.write("Enter the sensor values below:")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                label=feature,
                value=0.0,
                format="%.4f"
            )

    if st.button("Predict RUL", type="primary"):
        try:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            status, color, message = get_health_status(prediction)

            st.markdown("---")
            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted RUL", f"{prediction:.2f} cycles")
            with col2:
                st.markdown(f"**Health Status:** :{color}[{status}]")
            with col3:
                st.write(message)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Remaining Useful Life (cycles)"},
                gauge={
                    'axis': {'range': [0, 250]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcccc"},
                        {'range': [50, 100], 'color': "#ffe0b3"},
                        {'range': [100, 250], 'color': "#ccffcc"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("**Project:** Predictive Maintenance System using NASA C-MAPSS Dataset | Diploma Computer Engineering")