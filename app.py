from flask import Flask, render_template, request,redirect,url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
app = Flask(__name__)

# Load and prepare dataset
df = pd.read_csv("heart.csv")
X = df.drop(columns=['target'])
y = df['target']

# Feature Selection using RFE
model_rfe = LogisticRegression(max_iter=500)
rfe = RFE(model_rfe, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
labels = list(X.columns[rfe.support_])  # Selected feature names

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rfe)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)


# Train multiple models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "SVM": SVC(probability=True),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

accuracy_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_results[name] = accuracy_score(y_test, y_pred)

best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]

# Print model accuracy results in terminal
print("\nModel Accuracy Results:")
for name, acc in accuracy_results.items():
    print(f"{name}: {acc:.4f}")
print(f"\nBest Model: {best_model_name} with Accuracy: {accuracy_results[best_model_name]:.4f}")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_data = [float(request.form[label]) for label in labels]
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = best_model.predict(input_scaled)[0]
            prediction_result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"
            return redirect(url_for("result", prediction=prediction_result))
        except:
            return redirect(url_for("result", prediction="Invalid input values."))
    return render_template("index.html", labels=labels)

@app.route("/result")
def result():
    prediction = request.args.get("prediction", "No prediction available")
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)






