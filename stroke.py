
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_URL = "https://huggingface.co/srieas/TestModel/resolve/main/healthcare-dataset-stroke-data.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    return df


@st.cache_resource
def train_model():
    df = load_data()
    df_encoded = pd.get_dummies(df, drop_first=True, dtype=int)

    X = df_encoded.drop(columns=["id", "stroke"])
    y = df_encoded["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "stroke_recall": report.get("1", {}).get("recall", 0),
        "training_records": len(X_train),
        "testing_records": len(X_test),
    }

    return model, X.columns, metrics, y_test, y_pred


def build_patient_record():
    return {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "age": st.sidebar.slider("Age", 18, 100, 55),
        "hypertension": st.sidebar.checkbox("Hypertension"),
        "heart_disease": st.sidebar.checkbox("Heart disease"),
        "ever_married": st.sidebar.selectbox("Ever married", ["Yes", "No"]),
        "work_type": st.sidebar.selectbox(
            "Work type",
            ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
        ),
        "Residence_type": st.sidebar.selectbox("Residence type", ["Urban", "Rural"]),
        "avg_glucose_level": st.sidebar.slider("Average glucose level", 50, 300, 120),
        "bmi": st.sidebar.slider("BMI", 10.0, 70.0, 28.0),
        "smoking_status": st.sidebar.selectbox(
            "Smoking status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"],
        ),
    }


def predict_stroke(model, feature_columns, patient_data):
    record = pd.DataFrame([patient_data])
    encoded = pd.get_dummies(record, drop_first=True, dtype=int)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    probability = model.predict_proba(encoded)[0][1]
    prediction = model.predict(encoded)[0]
    return prediction, float(probability), encoded


def main():
    st.set_page_config(page_title="Stroke Risk Predictor", layout="wide")
    st.title("Real-Time Stroke Risk Prediction")
    st.caption("Live risk estimation using a stroke-risk decision tree model.")

    model, feature_columns, metrics, y_test, y_pred = train_model()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Stroke Recall", f"{metrics['stroke_recall'] * 100:.2f}%")
    col3.metric("Training Records", f"{metrics['training_records']}")
    col4.metric("Testing Records", f"{metrics['testing_records']}")

    patient_data = build_patient_record()

    if st.button("Predict Stroke Risk", use_container_width=True):
        prediction, probability, _ = predict_stroke(model, feature_columns, patient_data)
        risk_label = "High Risk" if prediction == 1 else "Low Risk"
        risk_color = "#ff4b4b" if prediction == 1 else "#00c853"

        st.subheader("Prediction Result")
        st.markdown(
            f"<div style='font-size: 30px; color: {risk_color}; font-weight: bold;'>"
            f"{risk_label} — {probability * 100:.2f}% risk</div>",
            unsafe_allow_html=True,
        )
        st.progress(min_value=0.0, max_value=1.0, value=probability)
        st.json(patient_data)

    st.subheader("Model Performance")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("Dataset Preview")
    st.dataframe(load_data().head(10), use_container_width=True)


if __name__ == "__main__":
    main()
