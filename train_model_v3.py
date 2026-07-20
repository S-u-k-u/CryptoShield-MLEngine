import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import joblib

print("Loading Behavioral Dataset for Anomaly Detection...")
df = pd.read_csv("crypto_behavioral_dataset.csv")

# For Isolation Forest, we train ONLY on the "normal" (Secure == 1) data to learn the healthy manifold.
# In a real academic setting, you'd want the model to learn what "good" behavior looks like
# so it can flag anything else as an anomaly. Let's use the whole dataset but evaluation focuses on anomalies.
# Actually, Isolation Forest handles mixed datasets but isolates anomalies (the minority).
# Let's train on predominantly secure data to simulate learning a baseline.
df_secure = df[df['Secure'] == 1]

X_train_raw = df_secure[['Sequence', 'Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time']]

print("Building Preprocessing Pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(token_pattern=r'(?u)[a-zA-Z0-9\.]+'), 'Sequence'),
        ('num', StandardScaler(), ['Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time'])
    ])

print("Building Anomaly Detection Pipeline (Layer 2)...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('anomaly_detector', IsolationForest(n_estimators=100, contamination=0.01, random_state=42))
])

print("Training Isolation Forest on Secure Baseline...")
pipeline.fit(X_train_raw)

# Now test it on insecure data to see if it flags it
df_insecure = df[df['Secure'] == 0].sample(10, random_state=42)
X_test_insecure = df_insecure[['Sequence', 'Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time']]

y_pred_insecure = pipeline.predict(X_test_insecure)
# In Isolation Forest: 1 = Inlier (Normal), -1 = Outlier (Anomaly)
print("Testing on 10 known insecure samples (Expected: -1 for anomalies):")
print(y_pred_insecure)

# For SHAP explainability, SHAP struggles with full Pipelines natively sometimes, 
# so we will use the TreeExplainer on the isolated IsolationForest model during inference.
joblib.dump(pipeline, 'crypto_anomaly_pipeline.pkl')
print("Model Pipeline saved to crypto_anomaly_pipeline.pkl! Ready for Academic Deployment.")
