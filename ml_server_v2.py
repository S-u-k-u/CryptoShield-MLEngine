from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load Behavioral Pipeline (Includes Preprocessor and RandomForest)
print("Loading Research-Grade ML Pipeline...")
try:
    pipeline = joblib.load('crypto_behavioral_pipeline.pkl')
    print("Pipeline loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Could not find crypto_behavioral_pipeline.pkl.")
    exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "BehavioralRandomForest"})

@app.route('/verify_behavior', methods=['POST'])
def verify_behavior():
    """
    Endpoint for MLBridge.classifyBehavior().
    Expects JSON: {
        "sequence": "KeyGen.init -> Cipher.init",
        "ciphertext_entropy": 7.85,
        "iv_length": 16,
        "iv_entropy": 7.99,
        "exec_time": 1.25
    }
    Returns JSON: {"confidence": 0.98} # Probability of being INSECURE
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Empty payload"}), 400
        
    try:
        # Convert incoming JSON into a DataFrame for the sklearn Pipeline
        df = pd.DataFrame([{
            'Sequence': data.get('sequence', 'Unknown'),
            'Ciphertext_Entropy': float(data.get('ciphertext_entropy', 0.0)),
            'IV_Length': int(data.get('iv_length', 0)),
            'IV_Entropy': float(data.get('iv_entropy', 0.0)),
            'Exec_Time': float(data.get('exec_time', 0.0))
        }])
        
        # Predict Probabilities
        probabilities = pipeline.predict_proba(df)[0]
        
        # The probability of class 0 (Insecure) is what MLBridge uses as "Violation Confidence"
        prob_insecure = float(probabilities[0]) 
        
        print(f"[CryptoShield ML] Evaluated Behavioral Profile -> Insecure Probability: {prob_insecure:.4f}")

        # Fix for Java parsing bug: We only return the confidence key to make 
        # string isolation via lastIndexOf(':') work flawlessly.
        return jsonify({
            "confidence": prob_insecure
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n=======================================================")
    print(" CryptoShield AI Behavioral Engine running on port 5001")
    print(" Listening for dynamic execution profiles...")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5001, debug=False)
