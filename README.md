# ✈️ Flight Price Prediction using AWS SageMaker

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aman6303-dummy-flight-price-prediction.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)

An End-to-End Machine Learning project that predicts flight prices based on travel details. This project demonstrates a hybrid workflow: **Data Collection and Preprocessing were performed locally**, while the **Model Training was offloaded to the cloud using AWS SageMaker** for scalability.

## 🔗 Live Demo
Check out the deployed application here:  
**[Flight Price Prediction App](https://aman6303-dummy-flight-price-prediction.streamlit.app/)**

---

## 📖 Project Overview

The goal of this project is to build a regression model to predict the price of a flight ticket. It allows users to input travel parameters (source, destination, airline, departure time, etc.) and receive an estimated fare.

This project focuses on MLOps practices by integrating cloud resources for heavy lifting (training) while keeping costs low by handling lightweight tasks (preprocessing) locally.

### Key Features
* **Hybrid Workflow:** Optimized cost and efficiency by splitting tasks between Local Machine and Cloud (AWS).
* **AWS SageMaker Integration:** leveraged SageMaker estimators for model training.
* **Interactive UI:** Built with Streamlit for easy user interaction.
* **Data Pipeline:** Robust preprocessing pipeline handling categorical encoding and feature scaling.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Joblib, Boto3 (AWS SDK)
* **Cloud Platform:** AWS SageMaker, AWS S3
* **Frontend:** Streamlit
* **Version Control:** Git & GitHub

---

## ⚙️ Architecture & Workflow

1.  **Data Collection:** Dataset gathered locally.
2.  **Preprocessing (Local):**
    * Cleaning null values.
    * Feature Engineering (Date/Time extraction).
    * Encoding Categorical variables (OneHot/Label Encoding).
    * *Processed data is then uploaded to an S3 Bucket.*
3.  **Model Training (AWS SageMaker):**
    * S3 bucket acts as the data source.
    * A Scikit-learn estimator is spun up on a SageMaker training instance.
    * The model artifact (`model.tar.gz`) is saved back to S3.
4.  **Deployment:**
    * The trained model is downloaded/loaded into the Streamlit app.
    * App hosted on Streamlit Cloud.
