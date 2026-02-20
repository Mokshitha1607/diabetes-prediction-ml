import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.title("🩺 Diabetes Prediction System")

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# User input
st.header("Enter Patient Details")

preg = st.number_input("Pregnancies", 0, 20)
glu = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 140)
skin = st.number_input("Skin Thickness", 0, 100)
ins = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Pregnancies": preg,
        "Glucose": glu,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": ins,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    result = model.predict(input_data)

    if result[0] == 1:
        st.error("⚠ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")