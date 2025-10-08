import streamlit as st
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import pickle
from PIL import Image

# Load HOG + SVM model
with open('cat_dog_svm_hog.pkl','rb') as f:
    model = pickle.load(f)

st.title("Cats vs Dogs Classifier (HOG + SVM)")

uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg","png"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    img_array = np.array(img)
    img_array = resize(img_array, (128,128))
    img_gray = rgb2gray(img_array)
    
    features = hog(img_gray, pixels_per_cell=(16,16), cells_per_block=(2,2))
    features = features.reshape(1,-1)
    
    pred = model.predict(features)
    prob = model.predict_proba(features)[0]
    
    label = "Dog 🐶" if pred[0]==1 else "Cat 🐱"
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: {prob[pred[0]]*100:.2f}%")
