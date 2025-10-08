import os
import random
import pickle
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Small dataset folder
# -------------------------------
small_data_dir = 'dataset_small'
categories = ['cat','dog']

X, y = [], []

# -------------------------------
# Extract HOG features (grayscale)
# -------------------------------
for label, category in enumerate(categories):
    folder = os.path.join(small_data_dir, category)
    files = os.listdir(folder)
    for f in files:
        if f.lower().endswith(('.jpg','.png')):
            img_path = os.path.join(folder,f)
            img = imread(img_path)
            img = resize(img,(128,128))
            img_gray = rgb2gray(img)  # convert to grayscale
            features = hog(img_gray, pixels_per_cell=(16,16), cells_per_block=(2,2))
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

if len(X)==0:
    raise Exception("No images found in dataset_small!")

# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train SVM
# -------------------------------
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy with HOG features: {accuracy_score(y_test,y_pred)*100:.2f}%")

# -------------------------------
# Save model
# -------------------------------
with open('cat_dog_svm_hog.pkl','wb') as f:
    pickle.dump(clf,f)

print("Model saved as cat_dog_svm_hog.pkl")
