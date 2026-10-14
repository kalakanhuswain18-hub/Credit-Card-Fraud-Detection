# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using supervised learning, model evaluation, and a FastAPI prediction service.

## 🚀 Project Overview

Credit card fraud detection is an imbalanced classification problem because fraudulent transactions represent only a very small proportion of all transactions.

This project explores transaction data, trains and evaluates supervised machine learning models, selects a classification threshold, and exposes the trained model through a REST API.

## ✨ Features

- Data understanding and preprocessing
- Handling highly imbalanced classification data
- Training and comparing multiple supervised ML models
- Model evaluation using Precision, Recall, F1-score, confusion matrix, and Average Precision
- Threshold tuning for fraud detection
- Model explainability using SHAP
- FastAPI endpoint for transaction predictions
- Interactive API documentation through Swagger UI

## 🧰 Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, LightGBM
- **Explainability:** SHAP
- **API:** FastAPI, Pydantic
- **Model Persistence:** Joblib
- **Notebook:** Jupyter Notebook

## 📊 Dataset

The project uses a credit card transaction dataset containing 284,807 transactions and 31 columns.

- `Time`: Time elapsed since the first transaction in the dataset
- `V1`–`V28`: Anonymized, PCA-transformed features
- `Amount`: Transaction amount
- `Class`: Target label (`0` = genuine, `1` = fraudulent)

The dataset is highly imbalanced, with 492 fraudulent transactions among 284,807 original records.

**Note:** The dataset contains anonymized features. The model's predictions should be treated as risk estimates, not definitive proof of fraud.

## 🤖 Machine Learning Workflow

1. Load and inspect the dataset.
2. Check data quality and remove duplicate records.
3. Split the data using stratification.
4. Train supervised classification models.
5. Compare model performance using suitable evaluation metrics.
6. Tune the classification threshold based on precision and recall.
7. Save the selected model and its metadata.
8. Load the model through FastAPI for predictions.

## 📈 Model Evaluation

Because fraud is rare, accuracy alone is not sufficient to evaluate the model. Precision, Recall, F1-score, Average Precision, and false-positive/false-negative counts are also important.

The classification threshold can be adjusted depending on the trade-off between detecting more fraud and reducing false alerts.

## ⚡ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/kalakanhuswain18-hub/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the API

```bash
python -m uvicorn api.main:app --reload
```

### 5. Open Swagger documentation

Visit:

`http://127.0.0.1:8000/docs`

Use the `/predict` endpoint to submit a transaction and receive a fraud probability and predicted class.

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
├── api/
│   └── main.py
├── models/
│   ├── lightgbm_froud_model.pkl
│   └── model_metadata.json
├── notebooks/
│   └── 01_data_understanding.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚠️ Limitations

- The dataset's anonymized features limit direct interpretation of individual variables.
- Fraud predictions can produce false positives and false negatives.
- Evaluation metrics may vary across data splits and model configurations.
- This project is for educational and demonstration purposes; it is not a production-validated financial fraud prevention system.

## 👨‍💻 Author

**Kalakanhu Swain**

GitHub: [kalakanhuswain18-hub](https://github.com/kalakanhuswain18-hub)