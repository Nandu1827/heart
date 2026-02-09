# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# app = Flask(
#     __name__,
#     template_folder="../frontend/templates",
#     static_folder="../frontend/static"
# )

 
# models = {
#     "adaboost": pickle.load(open("model/adaboost.pkl", "rb")),
#     "catboost": pickle.load(open("model/catboost.pkl", "rb")),
#     "xgboost": pickle.load(open("model/xgboost.pkl", "rb"))
# }

# @app.route("/")
# def home():
#     return render_template("index.html", pred_class=None)

# @app.route("/predict", methods=["POST"])
# def predict():
#     model_name = request.form.get("model", "adaboost")
#     model = models[model_name]

#     features = [
#         float(request.form["age"]),
#         float(request.form["sex"]),
#         float(request.form["cp"]),
#         float(request.form["trtbps"]),
#         float(request.form["chol"]),
#         float(request.form["fbs"]),
#         float(request.form["restecg"]),
#         float(request.form["thalachh"]),
#         float(request.form["exng"]),
#         float(request.form["oldpeak"]),
#         float(request.form["slp"]),
#         float(request.form["caa"]),
#         float(request.form["thall"])
#     ]

#     print("Received values:", features)

#     input_data = np.array(features).reshape(1, -1)
#     pred_class = model.predict(input_data)[0]

#     probability = model.predict_proba(input_data)[0][1] * 100
#     confidence = round(probability, 2)

#     if confidence < 40:
#         risk = "Low"
#     elif confidence < 70:
#         risk = "Moderate"
#     else:
#         risk = "High"

#     result = f"Output: {pred_class} | Result: {'Heart Disease' if probability >= 50 else 'No Heart Disease'} | Confidence: {probability:.2f}%"

#     return render_template("index.html",
#                        prediction_text=result,
#                        pred_class=pred_class, 
#                        confidence=confidence,
#                        risk=risk
#                        )

 
# if __name__ == "__main__":
#     app.run(debug=True)

# --------------------------------------------------------------------------------------------------------------

# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import os

# app = Flask(
#     __name__,
#     template_folder="../frontend/templates",
#     static_folder="../frontend/static"
# )

# # -------- Load Models (deployment-safe paths) --------
# BASE_DIR = os.path.dirname(__file__)

# models = {
#     "adaboost": pickle.load(open(os.path.join(BASE_DIR, "model", "adaboost.pkl"), "rb")),
#     "catboost": pickle.load(open(os.path.join(BASE_DIR, "model", "catboost.pkl"), "rb")),
#     "xgboost": pickle.load(open(os.path.join(BASE_DIR, "model", "xgboost.pkl"), "rb"))
# }

# # -------- Routes --------
# @app.route("/")
# def home():
#     return render_template("index.html", pred_class=None)

# @app.route("/predict", methods=["POST"])
# def predict():
#     model_name = request.form.get("model", "adaboost")
#     model = models[model_name]

#     features = [
#         float(request.form["age"]),
#         float(request.form["sex"]),
#         float(request.form["cp"]),
#         float(request.form["trtbps"]),
#         float(request.form["chol"]),
#         float(request.form["fbs"]),
#         float(request.form["restecg"]),
#         float(request.form["thalachh"]),
#         float(request.form["exng"]),
#         float(request.form["oldpeak"]),
#         float(request.form["slp"]),
#         float(request.form["caa"]),
#         float(request.form["thall"])
#     ]

#     input_data = np.array(features).reshape(1, -1)
#     pred_class = model.predict(input_data)[0]

#     probability = model.predict_proba(input_data)[0][1] * 100
#     confidence = round(probability, 2)

#     if confidence < 40:
#         risk = "Low"
#     elif confidence < 70:
#         risk = "Moderate"
#     else:
#         risk = "High"

#     result = f"Output: {pred_class} | Result: {'Heart Disease' if probability >= 50 else 'No Heart Disease'} | Confidence: {probability:.2f}%"

#     return render_template(
#         "index.html",
#         prediction_text=result,
#         pred_class=pred_class,
#         confidence=confidence,
#         risk=risk
#     )

# if __name__ == "__main__":
#     app.run(debug=True)

# ------------------------------------------------------------------------------------------------------------------------------------
from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import gdown

# ---------- Flask App ----------
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# ---------- Model Directory ----------
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Model File IDs ----------
files = {
    "adaboost": ("1jUCzeM-KFNaYjgYl2PkfEV_zD2szjAkE", "adaboost.pkl"),
    "catboost": ("1L29LWJFPcg3yUHmxj8ixNNpNjgKlF9jC", "catboost.pkl"),
    "xgboost": ("1toPVI2VjQLSv9HS_ajq9xJIH0ujPOpnz", "xgboost.pkl")
}

models = {}

# ---------- Lazy Load Function ----------
def load_model(model_name):
    if model_name in models:
        return models[model_name]

    file_id, filename = files[model_name]
    file_path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False, fuzzy=True)

    with open(file_path, "rb") as f:
        models[model_name] = pickle.load(f)

    return models[model_name]

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html", pred_class=None)

@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form.get("model", "adaboost")
    model = load_model(model_name)   # FIXED

    features = [
        float(request.form["age"]),
        float(request.form["sex"]),
        float(request.form["cp"]),
        float(request.form["trtbps"]),
        float(request.form["chol"]),
        float(request.form["fbs"]),
        float(request.form["restecg"]),
        float(request.form["thalachh"]),
        float(request.form["exng"]),
        float(request.form["oldpeak"]),
        float(request.form["slp"]),
        float(request.form["caa"]),
        float(request.form["thall"])
    ]

    input_data = np.array(features).reshape(1, -1)

    pred_class = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100
    confidence = round(probability, 2)

    if confidence < 40:
        risk = "Low"
    elif confidence < 70:
        risk = "Moderate"
    else:
        risk = "High"

    result = (
        f"Output: {pred_class} | "
        f"Result: {'Heart Disease' if probability >= 50 else 'No Heart Disease'} | "
        f"Confidence: {probability:.2f}%"
    )

    return render_template(
        "index.html",
        prediction_text=result,
        pred_class=pred_class,
        confidence=confidence,
        risk=risk
    )

# ---------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)