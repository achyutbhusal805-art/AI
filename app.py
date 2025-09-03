from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        pregnancies = float(request.form["pregnancies"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form["skin_thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        pedigree = float(request.form["pedigree"])
        age = float(request.form["age"])

        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                              insulin, bmi, pedigree, age]])
        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        result = "High Risk of Diabetes" if prediction == 1 else "Low Risk of Diabetes"
        return render_template("result.html", result=result, probability=round(prob*100, 2))
    return render_template("predict.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
