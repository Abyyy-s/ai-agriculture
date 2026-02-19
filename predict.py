import joblib
import pandas as pd

# load saved model
model = joblib.load("crop_model.pkl")

# example input
sample = pd.DataFrame(
    [[90,42,43,20,82,6.5,202]],
    columns=["N","P","K","temperature","humidity","ph","rainfall"]
)

prediction = model.predict(sample)

print("Recommended Crop:", prediction[0])
