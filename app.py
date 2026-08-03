import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🛠️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_artifacts():
    model = joblib.load("rul_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_artifacts()

# Friendly names for sensors
friendly_names = {
    "setting1": "Operating Setting 1",
    "setting2": "Operating Setting 2",
    "s2": "Fan Inlet Temperature",
    "s3": "LPC Outlet Temperature",
    "s4": "HPC Outlet Temperature",
    "s6": "Fan Inlet Pressure",
    "s7": "Bypass Duct Pressure",
    "s8": "HPC Outlet Pressure",
    "s9": "Physical Fan Speed",
    "s11": "Static Pressure at HPC Outlet",
    "s12": "Ratio of Fuel Flow to Ps30",
    "s13": "Corrected Fan Speed",
    "s14": "Corrected Core Speed",
    "s15": "Bypass Ratio",
    "s17": "Bleed Enthalpy",
    "s20": "High-Pressure Turbine Coolant Bleed",
    "s21": "Low-Pressure Turbine Coolant Bleed"
}

# Default realistic values
default_values = {
    "setting1": 0.0020,
    "setting2": 0.0000,
    "s2": 643.0,
    "s3": 1589.0,
    "s4": 1407.0,
    "s6": 21.60,
    "s7": 553.0,
    "s8": 2388.0,
    "s9": 9050.0,
    "s11": 47.30,
    "s12": 522.0,
    "s13": 2388.0,
    "s14": 8130.0,
    "s15": 8.42,
    "s17": 392.0,
    "s20": 38.90,
    "s21": 23.30
}

# Health Status function
def get_health_status(rul):
    if rul > 100:
        return "Healthy", "green", "✅ Engine is in good condition. No immediate action needed."
    elif rul > 50:
        return "Warning", "orange", "⚠️ Maintenance should be planned soon."
    else:
        return "Critical", "red", "🚨 Immediate maintenance is required!"

# ========== HEADER ==========
st.title("🛠️ AI-Powered Predictive Maintenance System")
st.markdown("### Predict Remaining Useful Life (RUL) of Turbofan Engines")
st.markdown("This system uses Machine Learning to predict how many more cycles an engine can safely run before maintenance is needed.")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Input Method", ["Manual Sensor Input", "Upload CSV File"])
st.sidebar.markdown("---")
st.sidebar.info("This project uses the NASA C-MAPSS Turbofan Engine dataset.")

# ========== MANUAL INPUT ==========
if option == "Manual Sensor Input":
    st.subheader("📊 Enter Current Sensor Readings")
    st.caption("Realistic example values are already filled. You can change them to see different predictions.")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(features):
        display_name = friendly_names.get(feature, feature)
        default = default_values.get(feature, 0.0)
        
        with cols[i % 3]:
            input_data[feature] = st.number_input(
                label=display_name,
                value=float(default),
                format="%.4f",
                key=feature
            )

    st.markdown("")
    predict_btn = st.button("🔍 Predict Remaining Useful Life", type="primary", use_container_width=True)

    if predict_btn:
        try:
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            status, color, message = get_health_status(prediction)

            st.markdown("---")
            st.subheader("📈 Prediction Result")

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Predicted RUL", value=f"{prediction:.1f} cycles")
            with col2:
                st.markdown(f"### Health Status")
                st.markdown(f":{color}[**{status}**]")
            with col3:
                st.markdown("### Recommendation")
                st.write(message)

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={'text': "Remaining Useful Life (cycles)", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 250], 'tickwidth': 1},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffcccc"},
                        {'range': [50, 100], 'color': "#ffe0b3"},
                        {'range': [100, 250], 'color': "#c6efce"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error while predicting: {e}")

# ========== CSV UPLOAD ==========
else:
    st.subheader("📁 Upload Sensor Data (CSV File)")
    st.info("Upload a CSV file containing sensor readings of one or more engines.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            missing_cols = [col for col in features if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
            else:
                X = df[features]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)

                result_df = df.copy()
                result_df["Predicted_RUL"] = np.round(predictions, 1)

                health_list = []
                for rul in predictions:
                    status, _, _ = get_health_status(rul)
                    health_list.append(status)
                result_df["Health_Status"] = health_list

                st.success("✅ Predictions completed successfully!")
                st.write("### Prediction Results")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="rul_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# Footer
st.markdown("---")
st.caption("Project: Predictive Maintenance System using NASA C-MAPSS Dataset | Diploma in Computer Engineering")
