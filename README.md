# 🐱🐶 Cats vs Dogs Classifier (HOG + SVM)

This project is a **Cats vs Dogs image classifier** built using **Histogram of Oriented Gradients (HOG) features** and a **Support Vector Machine (SVM)** classifier.  
It is designed to be lightweight, easy to train on small datasets, and provides accurate predictions for classifying images as either a cat 🐱 or a dog 🐶.

---

## 🌟 Project Overview

Image classification is a fundamental task in computer vision. The goal of this project is to **detect whether an uploaded image contains a cat or a dog**.  
Instead of using heavy deep learning models, this project leverages **HOG features** to capture the essential shapes and edges in the images, combined with a **Linear SVM** for classification.  

This approach is **computationally efficient** and works well even with a small dataset (50–100 images per class). It provides a fast and interpretable method for beginners to understand image classification techniques without relying on large pre-trained neural networks.

---

## 📈 Approach

1. **Dataset Preparation**  
   - A subset of the Kaggle Cats vs Dogs dataset is used.  
   - Images are organized into `cat/` and `dog/` folders for training.  
   - Only a small number of images are used to demonstrate the workflow efficiently.

2. **Feature Extraction**  
   - **Histogram of Oriented Gradients (HOG)** is applied to each image.  
   - HOG captures the **edge and shape information**, which is crucial for distinguishing cats and dogs.

3. **Classification**  
   - **Support Vector Machine (SVM)** with a linear kernel is trained on the extracted HOG features.  
   - SVM is chosen for its effectiveness in **high-dimensional feature spaces** and robustness with small datasets.

4. **Prediction and Deployment**  
   - A trained model (`cat_dog_svm_hog.pkl`) is used to make predictions on new images.  
   - The classifier is deployed via a **Streamlit web app** for interactive predictions, showing both the predicted class and confidence score.

---

## ⚡ Features

- **Lightweight and Efficient:** Works with a small dataset without requiring heavy computation.  
- **Interpretable:** Uses HOG features that are easy to understand visually.  
- **Interactive Web App:** Streamlit interface allows users to upload images and see predictions instantly.  
- **Confidence Score:** Each prediction includes a confidence level to show certainty.  
- **Extendable:** Can be scaled with larger datasets or replaced with deep learning models for improved accuracy.

---

## 🌐 Live Demo

You can try the classifier online via Streamlit:  
[**Open Streamlit App**](https://sctml3-lkdryvg4igycryb7c76kfb.streamlit.app/)  

---

## 🔍 Key Insights

- With only 50 images per class, the model achieves reasonable accuracy (~70–80%).  
- HOG + SVM is ideal for **quick experimentation** and learning the basics of image classification.  
- For **higher accuracy**, you can replace HOG with **pre-trained CNN features** like VGG16 or ResNet50.

---

## 📌 License

This project is open-source under the **MIT License**.


