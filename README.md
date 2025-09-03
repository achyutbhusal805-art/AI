# AI Diabetes Predictor WebApp

Flask-based web application that predicts diabetes risk using AI (XGBoost model).

## Features
- Multi-page layout (Home, About, Predict, Contact)
- AI-powered prediction using trained XGBoost model
- Professional Bootstrap design
- Ready for deployment to Render or Heroku

---

## 🚀 Deploy to Render (One Click)

Click the button below to deploy this app directly on [Render](https://render.com):

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Render settings:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

---

## 💻 Run Locally

Clone this repository and set it up on your computer:

```bash
git clone https://github.com/YOUR_USERNAME/diabetes_webapp_real.git
cd diabetes_webapp_real
pip install -r requirements.txt
```

Train the model (uses `diabetes.csv` included):
```bash
python train_model.py
```

Run the web app:
```bash
python app.py
```

Go to `http://127.0.0.1:5000` in your browser.

---

## ⚠️ Note
The included `diabetes.csv` is a **small demo dataset (5 rows)**.  
For realistic predictions, replace it with the **full PIMA dataset** from Kaggle and rerun `python train_model.py`.
