import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load dataset
df = pd.read_csv(r"C:\Users\User\OneDrive\Desktop\ICIKIWIR\gestures.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (KNN)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print(f"✅ Akurasi model: {acc:.2f}")

# Save model
joblib.dump(model, "gesture_model.pkl")
print("✅ Model tersimpan: gesture_model.pkl")