import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading Behavioral Dataset...")
df = pd.read_csv("crypto_behavioral_dataset.csv")

X = df[['Sequence', 'Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time']]
y = df['Secure']

print("Building Preprocessing Pipeline...")
# We must process text (Sequence) via TF-IDF, and scale the continuous variables.
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(token_pattern=r'(?u)[a-zA-Z0-9\.]+'), 'Sequence'),
        ('num', StandardScaler(), ['Ciphertext_Entropy', 'IV_Length', 'IV_Entropy', 'Exec_Time'])
    ])

print("Building Evaluation Pipeline (Layer 2 - Dynamic Behavioral Verification)...")
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("Training Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

print("Evaluating Behavioral Model...")
y_pred = pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, 'crypto_behavioral_pipeline.pkl')
print("Model Pipeline saved to crypto_behavioral_pipeline.pkl! Ready for Research-Grade API deployment.")
