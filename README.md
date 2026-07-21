# 🧠 CryptoShield: Machine Learning Engine

> **🚧 WORK IN PROGRESS:** This ML microservice is in its experimental phase as part of an ongoing university research project. The training datasets, model weights, and API responses are subject to frequent changes and ongoing tuning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-lightgrey?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange?logo=scikit-learn)

The **Machine Learning Engine** is a microservice designed to work alongside the [CryptoShield Java Enforcement Engine](https://github.com/S-u-k-u/CryptoShield). It provides research-grade behavioral classification (Layer 2 Verification) to detect insecure cryptographic API usage dynamically.

This repository serves as the Python-based backend that processes execution sequences, entropy calculations, and performance metrics to generate a **confidence score (0.0 to 1.0)** indicating the probability of crypto-misuse.

---

## 🎯 Key Features

- **Behavioral Classification**: Uses machine learning (Random Forest / TF-IDF pipelines) trained on CWE rules and MASCBench patterns to evaluate cryptographic runtime behavior.
- **Microservice Architecture**: Runs as an independent Flask API (default: `port 5001`), allowing the Java runtime agent to query it asynchronously.
- **Explainable AI (XAI) Ready**: Returns not just a confidence score, but a localized explanation indicating *why* a call was flagged (e.g., "IV_Entropy deviation").
- **Fallback Resiliency**: The CryptoShield Java engine is designed to seamlessly fall back to deterministic rule-based enforcement if this service goes offline.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### 2. Installation
Clone this repository and install the required machine learning dependencies.

```bash
git clone https://github.com/S-u-k-u/CryptoShield-MLEngine.git
cd CryptoShield-MLEngine

# (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install Flask scikit-learn numpy pandas
```

### 3. Running the Server

Start the classification service:
```bash
# Depending on the active version you are testing:
python ml_server.py
```
*The server will start on `http://127.0.0.1:5001/`.*

---

## 🔌 API Endpoints

The Java agent (`MLBridge.java`) automatically communicates with these endpoints.

### `GET /health`
Checks if the ML service is online.
- **Response:** `200 OK`

### `POST /verify_behavior`
Evaluates a cryptographic configuration vector.
- **Payload (JSON):**
  ```json
  {
      "sequence": "AES/CBC/PKCS5Padding...",
      "ciphertext_entropy": 3.42,
      "iv_length": 16,
      "iv_entropy": 1.05,
      "exec_time": 12.4
  }
  ```
- **Response (JSON):**
  ```json
  {
      "confidence": 0.98,
      "explanation": "Deviation detected: IV_Entropy critically low"
  }
  ```

---

## 🛠 Model Training Pipeline

The engine includes several iterations of model training scripts (`train_model.py`, `train_model_v4.py`) used to generate the `.pkl` files. 

If you wish to re-train the model with new behavioral datasets:
```bash
python generate_dataset_v2.py
python train_model_v4.py
```
This will output updated pipeline artifacts (`crypto_rf_model.pkl`, etc.) which are automatically loaded by the Flask server upon restart.

---

## 👥 Authors
- **Sanjay Venkat** (23BCE1928)
- **Sukumaran Sakthevelan** (23BCE1876)

*Semester End Project — Cryptography*
