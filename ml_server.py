from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load Model Artifacts
print("Loading ML Model into memory...")
try:
    model = joblib.load('crypto_rf_model.pkl')
    vectorizer = joblib.load('crypto_vectorizer.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Could not find model files. Did you run train_model.py?")
    exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for Java MLBridge.isAvailable() to ping."""
    return jsonify({"status": "online", "model": "RandomForestClassifier"})

@app.route('/classify', methods=['POST'])
def classify():
    """
    Endpoint for MLBridge.classify().
    Expects JSON: {"algorithm": "DES", "type": "CIPHER"}
    Returns JSON: {"algorithm": "DES", "secure": 0, "confidence": 0.99}
    """
    data = request.get_json()
    
    if not data or 'algorithm' not in data:
        return jsonify({"error": "Missing 'algorithm' in request body"}), 400
        
    algo_string = data['algorithm']
    
    # 1. Vectorize the input
    vectorized_input = vectorizer.transform([algo_string])
    
    # 2. Predict Classification
    # .predict() returns [1] or [0]
    prediction = int(model.predict(vectorized_input)[0])
    
    # 3. Get Confidence / Probability
    # .predict_proba() returns [[prob_0, prob_1]]
    probabilities = model.predict_proba(vectorized_input)[0]
    
    # The confidence is simply the highest probability
    # If the model predicts 0 (Insecure), the confidence is prob_0
    # If the model predicts 1 (Secure), the confidence is prob_1
    confidence = float(max(probabilities))
    
    # If the model predicts it's SECURE, the MLBridge logic expects confidence
    # that it is a *VIOLATION*. Thus, if safe, we shouldn't trigger high violation confidence.
    # The Java logic seems to expect high numbers to mean "High confidence it is a misuse/violation".
    # Let's adjust confidence to mean "Probability of being INSECURE" for MLBridge.
    prob_insecure = float(probabilities[0]) # probability of class 0 (Insecure)
    
    print(f"[CryptoShield ML] Evaluated '{algo_string}' -> Insecure Prob: {prob_insecure * 100:.1f}%")

    return jsonify({
        "algorithm": algo_string,
        "prediction": prediction,      # 1 = Secure, 0 = Insecure
        "confidence": prob_insecure    # The probability this is INSECURE
    })

if __name__ == '__main__':
    # Run the server on Port 5001 as expected by MLBridge.java
    print("\n=======================================================")
    print(" CryptoShield AI Engine is running on port 5001")
    print(" Waiting for cryptographic algorithms to classify...")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5001, debug=False)
