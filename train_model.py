import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading dataset...")
df = pd.read_csv("crypto_dataset.csv")

# We only care about the algorithm string and whether it's secure for this simple model
X = df['algorithm']
y = df['secure']

print("Vectorizing Text Data (TF-IDF)...")
# TF-IDF converts text like "AES/GCM" into mathematical arrays the model can understand.
# We split by '/' or '-' to understand components like 'AES' vs 'GCM' vs 'ECB'
vectorizer = TfidfVectorizer(token_pattern=r'(?u)[a-zA-Z0-9]+')
X_vectorized = vectorizer.fit_transform(X)

print("Splitting into Training and Testing Sets...")
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Evaluating Model Accuracy...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Saving Model and Vectorizer Artifacts...")
joblib.dump(model, 'crypto_rf_model.pkl')
joblib.dump(vectorizer, 'crypto_vectorizer.pkl')

print("SUCCESS: Model saved as crypto_rf_model.pkl. Ready for API deployment!")
