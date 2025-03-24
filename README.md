# Speech Emotion Recognition using Machine Learning

This project focuses on recognizing human emotions from speech audio using Machine Learning (ML) and Deep Learning (DL) techniques.

**MFCC feature** extraction is used and the model is trained using CNN, SVM, Random Forest, and XGBoost to classify emotions into Angry, Sad, Neutral, and Happy.

# 📌 Project Overview

# **Dataset**

Dataset Source - RAVDESS

In this project, I use [RAVDESS](https://zenodo.org/records/1188976#.Xl-poCEzZ0w) dataset to train.

You can find this dataset in kaggle or click on below link.

[Dataset for Speech Recognition](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

# Technologies & Libraries Used
**Python** – Core programming language for implementation

**TensorFlow & Keras** – Used to build and train the CNN model

**Librosa** – Used for audio processing and feature extraction (MFCC)

**Scikit-Learn** – Used for data preprocessing, encoding, and machine learning models (SVM, Random Forest)

**XGBoost** – Used for training an optimized XGBoost classifier

**Matplotlib & Seaborn** – Used for visualization of accuracy and confusion matrices

**NumPy & Pandas** – Used for handling datasets and numerical operations

**Joblib** – Used for saving and loading trained models

The RAVDESS dataset is a widely used benchmark for speech-based emotion recognition. It consists of 1,440 high-quality audio files featuring 24 professional actors (12 male, 12 female) expressing eight different emotions:

- Neutral (01)
- Calm (02)
- Happy (03)
- Sad (04)
- Angry (05)
- Fearful (06)
- Disgust (07)
- Surprised (08)

Each audio sample follows a structured naming convention, providing information about emotion, intensity, statement, repetition, and actor ID. The dataset includes speech at two intensity levels (normal and strong) and is recorded at 48kHz WAV format, ensuring high fidelity.

**Filename example**: 03-01-06-01-02-01-12.wav

Audio-only (03)

Speech (01)

Fearful (06)

Normal intensity (01)

Statement "dogs" (02)

1st Repetition (01)

12th Actor (12)

Female, as the actor ID number is even.

# Features 
Extracted MFCC features (40 coefficients per sample).

# Models Used
- Convolutional Neural Network (CNN)
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

# Model Selection 
The best-performing model is saved and used for testing.

# Steps to Implement
1️⃣ **Dataset Collection & Preprocessing**

The dataset contains audio files representing different emotions.

Filenames contain encoded emotion labels, which we map to 4 emotion classes:

- Angry (05, 07)
- Sad (04, 06)
- Neutral (01, 02)
- Happy (03, 08)

MFCC Feature Extraction is performed on each .wav file.

2️⃣ **Feature Extraction using MFCC**

- We extract MFCC (Mel-Frequency Cepstral Coefficients) features from each speech file using librosa.

- Mean pooling is applied to get a fixed-length feature vector (40 features).

```
import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)  # Take the mean across time
```

3️⃣ **Data Preprocessing**
- Convert Labels to One-Hot Encoding for classification.

- Split Data into training and testing sets.

- Apply Standard Scaling for SVM, Random Forest, and Neural Networks.

- Data Augmentation is applied by adding small Gaussian noise.

```
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Encode labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert to one-hot encoding
y_one_hot = to_categorical(y_encoded, num_classes=4)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Standard Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

4️⃣ **Model Training**

We implemented and trained four different models:

**1. CNN Model**

- Conv1D layers with Batch Normalization for feature extraction.

- Dropout Regularization to prevent overfitting.

- Softmax Activation for classification.
```
from tensorflow.keras import layers, models

cnn_model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

**2. Support Vector Machine (SVM)**
  
- Kernel: Radial Basis Function ('rbf').

- Hyperparameter Tuning using GridSearchCV.
```
from sklearn.svm import SVC
svm_model = SVC(kernel='rbf', C=10, gamma='scale')
svm_model.fit(X_train, y_train.argmax(axis=1))
```

**3. Random Forest Classifier**

- Trained using 100 decision trees (n_estimators=100).

- Feature importance analysis for better accuracy.
```
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train.argmax(axis=1))
```

**4. XGBoost Classifier**

- Optimized using GridSearchCV.

- Learning rate and depth tuning.
```
from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train.argmax(axis=1))

```
5️⃣ **Model Evaluation & Selection**

We compare model performances and save the best-performing model
```
model_performance = {
    "CNN": cnn_accuracy,
    "SVM": svm_acc,
    "Random_Forest": rf_acc,
    "XGBoost": xgb_acc
}

best_model_name = max(model_performance, key=model_performance.get)

if best_model_name == "CNN":
    cnn_model.save("best_model.h5")
else:
    import joblib
    joblib.dump(eval(best_model_name.lower() + "_model"), "best_model.pkl")
```

6️⃣ **Testing with New Audio Files**

- Load the saved model and test with a new .wav file.

- Extract MFCC features, reshape, and predict emotion.

📌 # Results
## 📊 Model Performance Comparison

| **Model**          | **Accuracy** |
|--------------------|-------------|
| **CNN**           | 82%  |
| **SVM**           | 48%    |
| **Random Forest** | 74%    |
| **XGBoost**       | 75%    |

**Best Model:** CNN (82%)

# Installation & Usage
**Install Dependencies**
```
pip install librosa numpy pandas scikit-learn tensorflow matplotlib xgboost joblib
```
# Author
👨‍💻 Nabila Rakib

📧 [Email me](nabila.rakib7@gmail.com)
🔗 Github:[Rakib007Nabila](https://github.com/Rakib007Nabila)

🚀 If you found this useful, don't forget to ⭐ the repo! 😊🎤

