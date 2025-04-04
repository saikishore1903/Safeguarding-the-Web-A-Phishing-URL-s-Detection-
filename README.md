# Safeguarding-the-Web-A-Phishing-URL-s-Detection-
This project is focused on detecting **phishing URLs** using various supervised **Machine Learning models** and providing an **interactive Streamlit web app** for real-time predictions. The application takes a URL input from the user and predicts whether it is **Phishing** or **Legitimate**, based on URL features. The solution incorporates data preprocessing, feature engineering, model training and evaluation, and finally, deployment through a well-structured UI.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training & Evaluation](#model-training--evaluation)
- [Streamlit Web App](#streamlit-web-app)
- [How to Use](#how-to-use)
- [Screenshots](#screenshots)
- [Conclusion](#conclusion)

---

## 🧠 Introduction

Phishing attacks are one of the most common cybersecurity threats. Cybercriminals often use fake websites to deceive users into revealing sensitive information. This project leverages the power of machine learning to identify such phishing URLs by analyzing specific characteristics within them. Multiple ML algorithms are trained and compared to provide accurate and reliable detection, supported by an intuitive Streamlit-based user interface.

---

## 📊 Dataset

The dataset used consists of labeled URL records, classified as either `phishing` or `legitimate`. Each URL is associated with various characteristics (features) that help in distinguishing between harmful and safe links.

- 📁 Data is split into **training** and **testing** sets (e.g., 70:30 ratio).
- 📊 Dataset distribution is visualized through a **pie chart** in the web app to show the proportion of phishing vs legitimate URLs.

---

## 🔧 Preprocessing

The raw data undergoes several preprocessing steps:

- **Label Encoding**: Convert target classes (Phishing / Legitimate) into numerical format.
- **Cleaning**: Drop unnecessary columns like indexes, timestamps, or unnamed entries.
- **Handling Missing Values**: Remove or impute incomplete records to ensure model stability.
- **Normalization** (optional): Applied for models sensitive to feature scale, like KNN or SVM.

---

## 🚀 Feature Engineering

Feature engineering plays a crucial role in phishing detection. Key features are extracted from each URL using regex and string operations:

| Feature | Description |
|--------|-------------|
| URL Length | Total number of characters in the URL |
| Use of `@` Symbol | Indicates redirection and suspicious behavior |
| Use of Hyphens (`-`) | Often used in fake domains |
| Presence of IP Address | If IP is used instead of domain name |
| Number of Dots (`.`) | Higher subdomain count can be suspicious |
| HTTPS Usage | Secure protocol presence |
| Presence of Suspicious Keywords | e.g., "login", "verify", "update" |
| Use of URL Shorteners | Services like bit.ly or tinyurl |
| Subdomain Count | Number of subdomains used |
| Special Characters | Unusual characters like `%`, `=`, etc. |

These features are engineered into numerical format and fed into the machine learning models.

---

## 🤖 Model Training & Evaluation

The following supervised ML models are trained and evaluated:

- ✅ Decision Tree
- ✅ XGBoost
- ✅ Support Vector Machine (SVM)
- ✅ K-Nearest Neighbors (KNN)

Each model is trained using the processed features and evaluated using the **testing set**. The following metrics are calculated and displayed in the app:

- 🎯 **Accuracy**
- 📈 **Precision**
- 📉 **Recall**
- 🧮 **F1-score**
- 🧾 **Confusion Matrix**: True Positives, True Negatives, False Positives, False Negatives
- 📊 **Evaluation Plots**: Metric comparisons across models

All trained models can be saved using `joblib` or `pickle` to avoid retraining during prediction.

---

## 🌐 Streamlit Web App

The application is built using **Streamlit** to provide an interactive and user-friendly interface.

### ✅ Key Features of the App:

1. **Dataset Overview**:
   - Pie chart showing the distribution of phishing and legitimate URLs.
   - Total number of records displayed.
2. **Feature Summary**:
   - Display of all the extracted features from each URL.
3. **Model Evaluation**:
   - Selection of any trained ML model from dropdown.
   - Displays corresponding confusion matrix and metrics like accuracy, precision, and recall.
   - Plots of performance metrics for visual comparison.
4. **Real-Time URL Prediction**:
   - User can input any URL into the textbox.
   - The selected model predicts whether it's *Phishing* or *Legitimate*.
   - Output is shown with labels and confidence indicators.

---
