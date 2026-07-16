import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())


# Encode target column
encoder = LabelEncoder()
df["Result"] = encoder.fit_transform(df["Result"])


# Split features and target
X = df.drop("Result", axis=1)
y = df["Result"]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create SVM model
model = SVC(
    kernel="linear",
    probability=True
)


# Train model
model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Model evaluation
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Save encoder
pickle.dump(encoder, open("encoder.pkl", "wb"))


print("Model, Scaler and Encoder saved successfully!")