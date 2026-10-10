# 💳 Credit Risk Analytics — Finance Data Mining App

**Data Mining Lab • Assignment #2**  
Interactive Streamlit application for **Loan Default Prediction** with advanced Feature Engineering, 3D visuals, and production-style frontend.

---

## Highlights

- **Domain:** Finance / Credit Risk
- **Login-protected** modern dashboard
- **Full pipeline:** Load → Clean → Feature Engineering → EDA → Train → Predict
- **3D Interactive Visuals** (Plotly scatter_3d — rotate, zoom, hover)
- **Rich Feature Engineering** (ratios, risk flags, composite score, log transforms)
- **Multiple Models:** Random Forest, Gradient Boosting, HistGradientBoosting, Logistic Regression
- **Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC + Confusion Matrix + ROC Curve
- **Live Prediction** form with risk factor explanation
- Premium dark finance-themed UI (not basic)

---

## Login Credentials

| Field    | Value         |
|----------|---------------|
| Username | `admin`       |
| Password | `finance123`  |

---

## Project Structure

```
finance_datamining_app/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
└── data/
    └── loan_default.csv
```

---

## Run Locally

```bash
# 1. Extract ZIP and enter folder
cd finance_datamining_app

# 2. (Optional but recommended) Virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# 3. Install
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Open http://localhost:8501

---

## Deploy on Streamlit Community Cloud (Free)

### Step-by-step:

1. Create a **GitHub** account (if needed) → https://github.com

2. Create a new **Public** repository  
   Example name: `credit-risk-datamining`

3. Upload **all files** from this project (including the `data/` folder and `.streamlit/` folder) into the repository.

   You can drag-and-drop on GitHub website or use:
   ```bash
   git init
   git add .
   git commit -m "Credit Risk Data Mining App"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/credit-risk-datamining.git
   git push -u origin main
   ```

4. Go to **https://share.streamlit.io** and sign in with GitHub.

5. Click **New app**
   - Repository → select your repo
   - Branch → `main`
   - Main file path → `app.py`
   - Click **Deploy**

6. Wait 1–3 minutes. Your live URL will look like:
   ```
   https://credit-risk-datamining-xxxxx.streamlit.app
   ```

7. Share this link + screenshots for assignment submission.

---

## Recommended Screenshots for Submission

- Login page
- Executive Dashboard (with 3D plot)
- Feature Engineering tab (new features table)
- 3D Scatter in EDA
- Model metrics + Confusion Matrix + ROC Curve + Feature Importance
- Live Prediction result (High Risk / Low Risk)

---

## Observations (for Q2 Documentation)

- Engineered features (`engineered_risk_score`, `loan_to_income_ratio`, late-payment flags) rank high in feature importance.
- Tree-based models achieve stronger separation than logistic regression on this non-linear credit data.
- 3D visualizations clearly show clustering of high-risk applicants in low credit-score / high DTI regions.
- After proper preprocessing + FE, models reach solid Accuracy / F1 / AUC suitable for the assignment.

---

**Disclaimer:** Academic use only. Not intended for real credit decisions.
