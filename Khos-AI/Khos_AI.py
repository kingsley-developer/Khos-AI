import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
import joblib

le = LabelEncoder()

df = pd.read_csv("healthcare_dataset.csv")

df["Blood Type"] = le.fit_transform(df["Blood Type"])

df["Medical Condition"] = le.fit_transform(df["Medical Condition"])

df["Medication"] = le.fit_transform(df["Medication"])

df["Admission Type"] = le.fit_transform(df["Admission Type"])

df["Test Results"] = le.fit_transform(df["Admission Type"])

df = df.dropna()

df = df.replace([np.inf, -np.inf],np.nan).dropna()

features = df[["Age", "Blood Type", "Medical Condition", "Medication", "Admission Type"]]

labels = df["Test Results"]

labels_classifier_out = df["Test Results"].nunique()


x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

kingsley_model = RandomForestClassifier()

kingsley_model.fit(x_train, y_train)

predictions = kingsley_model.predict(x_test)

mse = mean_squared_error(y_test, predictions)

joblib.dump(kingsley_model, "kingsley_hospital_model.joblib")

loaded_model = joblib.load("kingsley_hospital_model.joblib")

y_pred_loaded = loaded_model.predict(x_test)

accuracy_loaded = accuracy_score(y_test, y_pred_loaded)

print("Khos_AI model accuracy: ", accuracy_loaded)
print(f'Khos_AI model Mean Squared Error: {mse}')
print("Khos_AI model predictions: ",y_pred_loaded)

l = y_pred_loaded.ravel().tolist()# EXPAND THE ARRAY

print(l)