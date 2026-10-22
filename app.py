import streamlit as st
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Heart Disease Prediction",

)

st.title("Heart Disease Prediction System")
st.write("Machine Learning project using exactly four input attributes.")

folder = Path(__file__).parent
data_file = folder / "processed_cleveland.data"
model_file = folder / "heart_disease_model.pkl"

@st.cache_resource
def load_or_train_model():
    if model_file.exists():
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # Verify that the saved model uses the expected four inputs.
        if list(model.feature_names_in_) != [
            "age", "sex", "cp", "thalach"
        ]:
            raise ValueError("The saved model does not use the expected four features.")

        return model

    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "num"
    ]

    df = pd.read_csv(
        data_file, header=None, names=columns, na_values="?"
    )

    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["num"]).copy()
    df["heart_disease"] = (df["num"] > 0).astype(int)

    features = ["age", "sex", "cp", "thalach"]
    X = df[features]
    y = df["heart_disease"]

    preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), ["age", "thalach"]),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), ["sex", "cp"])
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=4, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    return model

try:
    model = load_or_train_model()
except Exception as e:
    st.error(f"Unable to load or train the model: {e}")
    st.info("Check that the dataset and trained model are uploaded to GitHub.")
    st.stop()

st.success("Model is ready.")

st.subheader("Enter Patient Attributes")

age = st.slider("Age (years)", 20, 90, 55)

sex_option = st.selectbox("Sex", ["Female (0)", "Male (1)"])
sex = 0 if sex_option == "Female (0)" else 1

cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])

thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

if st.button("Predict"):
    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "thalach": thalach
    }])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    if prediction == 1:
        st.warning("Model prediction: Heart disease present")
    else:
        st.success("Model prediction: No heart disease detected")

    st.write(
        f"Estimated model probability for the positive class: "
        f"{probability * 100:.2f}%"
    )

st.caption(
    "Educational project only. This prediction is not a medical diagnosis."
)
