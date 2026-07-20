from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap
import numpy as np

# Import the custom transformer so joblib can deserialize it
from train_model_v4 import MarkovChainTransformer 

app = Flask(__name__)

print("Loading Conference-Grade Anomaly Detection Pipeline (Markov Chains + AUPRC)...")
try:
    pipeline = joblib.load('crypto_anomaly_markov_pipeline.pkl')
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['anomaly_detector']
    print("Pipeline loaded successfully.")
    
    # Initialize SHAP explainer for the Isolation Forest
    explainer = shap.TreeExplainer(model)
except FileNotFoundError:
    print("FATAL ERROR: Could not find crypto_anomaly_markov_pipeline.pkl.")
    exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model": "IsolationForest+Markov+SHAP"})

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
        
        prediction = pipeline.predict(df)[0]
        score = float(pipeline.decision_function(df)[0])
        
        if prediction == -1:
            conf = min(0.99, 0.60 + (abs(score) * 3.0)) # Maps anomalies appropriately to 60%-99%
        else:
            conf = max(0.01, 0.40 - (abs(score) * 2.0)) 
            
        prob_insecure = float(conf)
        
        # SHAP Explainability on discrete graphs and matrices
        X_transformed = preprocessor.transform(df)
        
        try:
            # Extract names from Markov Chain graph and Scaler
            markov_features = preprocessor.named_transformers_['markov'].get_feature_names_out()
            num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names = list(markov_features) + list(num_features)
        except:
            feature_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
            
        shap_values = explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            sample_shap = shap_values[0][0] 
        else:
            if len(shap_values.shape) > 1:
                sample_shap = shap_values[0]
            else:
                sample_shap = shap_values
                
        min_idx = np.argmin(sample_shap)
        most_anomalous_feature = feature_names[min_idx]
        most_anomalous_value = sample_shap[min_idx]
        
        explanation = f"{most_anomalous_feature} deviation (score {most_anomalous_value:.3f})"
        
        print(f"[CryptoShield ML] Evaluated Behavioral Profile -> Insecure Probability: {prob_insecure:.4f} | Reason: {explanation}")

        return jsonify({
            "confidence": prob_insecure,
            "explanation": explanation
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    print("\n=========================================================================")
    print(" CryptoShield Academic AI Engine (Markov Graph + Isolation Forest + SHAP)")
    print(" Running on port 5001 [Production WSGI Server]")
    print("=========================================================================\n")
    serve(app, host='127.0.0.1', port=5001)
