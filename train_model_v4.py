import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve
import joblib

# 1. Custom Markov Chain Transformer
# Mathematical formalization of the API Call Sequence Graph
class MarkovChainTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.state_space = []
        self.transition_matrix_size = 0
        
    def fit(self, X, y=None):
        # Determine unique API states to build the NxN adjacency matrix
        unique_states = set()
        for sequence in X:
            calls = [c.strip() for c in sequence.split('->')]
            unique_states.update(calls)
            
        self.state_space = sorted(list(unique_states))
        self.transition_matrix_size = len(self.state_space)
        return self

    def transform(self, X, y=None):
        # Convert each sequence into a flattened Transition Probability Matrix (Markov Graph)
        n_samples = len(X)
        n_states = self.transition_matrix_size
        X_transformed = np.zeros((n_samples, n_states * n_states))
        
        state_to_idx = {state: i for i, state in enumerate(self.state_space)}
        
        for i, sequence in enumerate(X):
            calls = [c.strip() for c in sequence.split('->')]
            if len(calls) < 2:
                # E.g., standalone SecureRandom.nextBytes
                if len(calls) == 1 and calls[0] in state_to_idx:
                    idx = state_to_idx[calls[0]]
                    X_transformed[i, idx * n_states + idx] = 1.0 # Self-loop
                continue
                
            # Populate the Directed Acyclic Graph transitions
            counts = np.zeros((n_states, n_states))
            for j in range(len(calls) - 1):
                s1 = calls[j]
                s2 = calls[j+1]
                if s1 in state_to_idx and s2 in state_to_idx:
                    idx1 = state_to_idx[s1]
                    idx2 = state_to_idx[s2]
                    counts[idx1, idx2] += 1
            
            # Normalize into Markov transition probabilities
            for row in range(n_states):
                row_sum = np.sum(counts[row, :])
                if row_sum > 0:
                    counts[row, :] = counts[row, :] / row_sum
                    
            X_transformed[i, :] = counts.flatten()
            
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        # Required for SHAP integration
        names = []
        for s1 in self.state_space:
            for s2 in self.state_space:
                names.append(f"Transition_{s1}_to_{s2}")
        return np.array(names)


# 2. Pipeline Execution
if __name__ == "__main__":
    print("Loading Behavioral Dataset for Structural Anomaly Detection...")
    df = pd.read_csv("crypto_behavioral_dataset.csv")

    # In isolation forests for cybersecurity, we build the manifold on Secure data only
    df_secure = df[df['Secure'] == 1]
    X_train_clean = df_secure[['Sequence', 'Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time']]

    print("Building Preprocessing Pipeline with Markov Chain Graph Theory...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('markov', MarkovChainTransformer(), 'Sequence'), # Replace TF-IDF with Discrete Math
            ('num', StandardScaler(), ['Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time'])
        ])

    print("Building Academic Anomaly Detection Pipeline...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Contamination is set slightly higher here as real-world security requires high sensitivity
        ('anomaly_detector', IsolationForest(n_estimators=100, contamination=0.05, random_state=42))
    ])

    print("Training Isolation Forest on the Secure Baseline Manifold...")
    pipeline.fit(X_train_clean)


    # 3. Academic Benchmarking Metrics (AUPRC)
    print("\n--- Evaluating Model using AUPRC (Area Under Precision-Recall Curve) ---")
    
    # Create an imbalanced test set (90% secure, 10% injected anomalies) explicitly for evaluation
    df_test_secure = df[df['Secure'] == 1].sample(1000, random_state=42)
    df_test_insecure = df[df['Secure'] == 0].sample(100, random_state=42)
    
    df_eval = pd.concat([df_test_secure, df_test_insecure])
    X_eval = df_eval[['Sequence', 'Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time']]
    
    # Ground truth: 1 for Anomaly (Insecure), 0 for Normal (Secure) 
    # This is reversed from the dataset to match standard AUPRC convention where the positive class is the anomaly
    y_eval_true = (df_eval['Secure'] == 0).astype(int) 

    # In Isolation Forest, decision_function returns lower scores for anomalies
    # To use standard AUPRC where higher score = more likely anomaly, we simply negate the score
    y_eval_scores = -pipeline.decision_function(X_eval) 

    # Calculate Academic Metric
    auprc = average_precision_score(y_eval_true, y_eval_scores)
    
    print(f"AUPRC (Area Under the Precision-Recall Curve): {auprc:.4f}")
    
    if auprc > 0.90:
        print("✓ AUPRC exceeds academic baseline (>0.90) for highly imbalanced anomaly detection.")
    else:
        print("⚠ Warning: AUPRC lacks precision for conference submission.")

    joblib.dump(pipeline, 'crypto_anomaly_markov_pipeline.pkl')
    print("Conference-Grade Model Pipeline saved to crypto_anomaly_markov_pipeline.pkl")
