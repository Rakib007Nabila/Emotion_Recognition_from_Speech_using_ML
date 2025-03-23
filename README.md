# Speech Emotion Recognition using Machine Learning

This project focuses on recognizing human emotions from speech audio using Machine Learning (ML) and Deep Learning (DL) techniques.

We use **MFCC feature** extraction and train models including CNN, SVM, Random Forest, and XGBoost to classify emotions into Angry, Sad, Neutral, and Happy.

# 📌 Project Overview

# **Dataset:**

Dataset Source - RAVDESS

In this project, I use [RAVDESS](RAVDESS) dataset to train.

You can find this dataset in kaggle or click on below link.
[Dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

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

# Features: 
Extracted MFCC features (40 coefficients per sample).

# Models Used:
- Convolutional Neural Network (CNN)
- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

# Model Selection: 
The best-performing model is saved and used for testing.

