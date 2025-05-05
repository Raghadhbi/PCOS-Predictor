
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("pcos_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def classify(prediction):
    return "PCOS Detected" if prediction == 1 else "No PCOS Detected"

def main():
    st.title("PCOS Prediction App")

    html_temp = """
    <div style="background-color:#FF4B4B;padding:10px">
    <h2 style="color:white;text-align:center;">PCOS Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.slider("Age", 15, 45, 25)
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    menstrual = st.selectbox("Menstrual Irregularity", ["Yes", "No"])
    menstrual_val = 1 if menstrual == "Yes" else 0
    testosterone = st.slider("Testosterone Level", 10.0, 100.0, 50.0)
    follicle = st.slider("Follicle Count", 0, 40, 20)

    inputs = np.array([[age, bmi, menstrual_val, testosterone, follicle]])
    inputs_scaled = scaler.transform(inputs)

    if st.button("Predict"):
        prediction = model.predict(inputs_scaled)
        result = classify(prediction[0])
        st.success(result)

if __name__ == "__main__":
    main()
