from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap
import numpy as np

app = Flask(__name__)

print("Loading Research-Grade Anomaly Detection Pipeline...")
try:
    pipeline = joblib.load('crypto_anomaly_pipeline.pkl')
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['anomaly_detector']
    print("Pipeline loaded successfully.")
    
    # Initialize SHAP explainer for the Isolation Forest
    explainer = shap.TreeExplainer(model)
except FileNotFoundError:
    print("FATAL ERROR: Could not find crypto_anomaly_pipeline.pkl.")
    exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "IsolationForest+SHAP"})

@app.route('/verify_behavior', methods=['POST'])
def verify_behavior():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Empty payload"}), 400
        
    try:
        df = pd.DataFrame([{
            'Sequence': data.get('sequence', 'Unknown'),
            'Ciphertext_Entropy': float(data.get('ciphertext_entropy', 0.0)),
            'IV_Length': int(data.get('iv_length', 0)),
            'IV_Entropy': float(data.get('iv_entropy', 0.0)),
            'Exec_Time': float(data.get('exec_time', 0.0))
        }])
        
        # 1. Pipeline Prediction (1 = Normal, -1 = Anomaly)
        prediction = pipeline.predict(df)[0]
        
        # Isolation Forest decision_function: lower score = more anomalous
        score = float(pipeline.decision_function(df)[0])
        
        # Determine confidence of anomaly. If prediction == -1, it's an anomaly.
        # We'll map the decision_function score to a 0.0 - 1.0 confidence of being an anomaly.
        # A highly negative score means high confidence of anomaly.
        if prediction == -1:
            # Scale score loosely. -0.2 might be very anomalous
            conf = min(1.0, max(0.5, abs(score) * 5.0))
        else:
            conf = max(0.0, 0.5 - score) # low anomaly confidence
            
        prob_insecure = float(conf)
        
        # 2. SHAP Explainability (XAI)
        # Transform the single sample using the preprocessor to get the feature matrix
        X_transformed = preprocessor.transform(df)
        
        # Get feature names from the preprocessor to map shap values back
        try:
            # Extract names from TF-IDF and Scaler
            text_features = preprocessor.named_transformers_['text'].get_feature_names_out()
            num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names = list(text_features) + list(num_features)
        except:
            feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
            
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_transformed)
        
        # For IsolationForest, shap_values is a 2D array for the single sample [shap_val_f1, shap_val_f2...]
        if isinstance(shap_values, list):
            sample_shap = shap_values[0][0] # Handle different shape returns
        else:
            if len(shap_values.shape) > 1:
                sample_shap = shap_values[0]
            else:
                sample_shap = shap_values
                
        # Find the feature that contributed the most to the ANOMALY (most negative SHAP value)
        min_idx = np.argmin(sample_shap)
        most_anomalous_feature = feature_names[min_idx]
        most_anomalous_value = sample_shap[min_idx]
        
        explanation = f"{most_anomalous_feature} deviation (score {most_anomalous_value:.3f})"
        
        print(f"[CryptoShield ML] Evaluated Behavioral Profile -> Insecure Probability: {prob_insecure:.4f} | Reason: {explanation}")

        # Return both confidence and the SHAP explanation
        return jsonify({
            "confidence": prob_insecure,
            "explanation": explanation
        })
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n=======================================================")
    print(" CryptoShield Academic AI Engine (Isolation Forest + SHAP)")
    print(" Running on port 5001")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5001, debug=False)
