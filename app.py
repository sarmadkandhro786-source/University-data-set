from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("models/gpa_model.pkl")

@app.route("/")
def home():
    return "GPA Prediction API Running Successfully"

@app.route("/test")
def test():
    return "Test Working"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    input_data = pd.DataFrame({
        "Attendance": [data["Attendance"]],
        "Assignments": [data["Assignments"]],
        "QuizScores": [data["QuizScores"]],
        "PreviousGPA": [data["PreviousGPA"]],
        "ExamResults": [data["ExamResults"]]
    })

    prediction = model.predict(input_data)[0]

    return jsonify({
        "Predicted GPA": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)