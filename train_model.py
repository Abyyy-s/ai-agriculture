import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv("Crop_recommendation.csv")

# inputs and output
X = data.drop("label", axis=1)
y = data["label"]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create model
model = RandomForestClassifier()

# train using training data
model.fit(X_train, y_train)

# predict test data
predictions = model.predict(X_test)

# check accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
import joblib

# save model
joblib.dump(model, "crop_model.pkl")

print("✅ Model saved successfully!")
