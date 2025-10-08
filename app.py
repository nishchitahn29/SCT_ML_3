import streamlit as st
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Title
st.title("🐾 Cat vs Dog Classifier using Support Vector Machine (SVM)")

# Load images
def load_data(data_dir):
    X, y = [], []
    for label, folder in enumerate(["cat", "dog"]):
        folder_path = os.path.join(data_dir, folder)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).resize((64, 64)).convert("L")
            X.append(np.array(img).flatten())
            y.append(label)
    return np.array(X), np.array(y)

# Data path
data_dir = "sample_data"
if os.path.exists(data_dir):
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='linear', C=1)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)

    st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

    uploaded_file = st.file_uploader("Upload an image to classify:", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).resize((64, 64)).convert("L")
        img_array = np.array(img).flatten().reshape(1, -1)
        img_scaled = scaler.transform(img_array)
        pred = model.predict(img_scaled)[0]
        label = "🐱 Cat" if pred == 0 else "🐶 Dog"

        st.image(uploaded_file, caption=f"Prediction: {label}", use_container_width=True)
else:
    st.warning("Please add sample data under 'sample_data/cat' and 'sample_data/dog' folders.")
