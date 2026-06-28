# 🏦 Loan Approval Prediction System

A Machine Learning web application that predicts whether a loan application is likely to be approved based on applicant information.

Built with:
- Python
- Scikit-learn
- Streamlit
- Pandas
- Joblib

---

## 📌 Project Overview

This project uses historical loan application data to train a machine learning model capable of predicting loan approval outcomes.

The application provides:

- Data preprocessing pipeline
- Machine Learning model training
- Model evaluation and comparison
- Interactive Streamlit dashboard
- Real-time loan approval predictions

---

## 🎯 Problem Statement

Financial institutions receive large numbers of loan applications. Manual evaluation can be time-consuming and inconsistent.

This system helps automate the decision-making process by analyzing applicant information and predicting whether a loan is likely to be approved.

---

## 📊 Dataset Features

| Feature | Description |
|----------|------------|
| Gender | Applicant gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Education level |
| Self_Employed | Employment status |
| ApplicantIncome | Applicant monthly income |
| CoapplicantIncome | Co-applicant monthly income |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment duration |
| Credit_History | Credit history record |
| Property_Area | Residential area |
| Loan_Status | Target variable |

---

## 🧠 Machine Learning Pipeline

### Data Preprocessing

- Missing value handling
- One-Hot Encoding for categorical variables
- Standard Scaling for numerical variables
- Train/Test Split

### Models Evaluated

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Evaluation Metrics

- Accuracy
- Classification Report
- Confusion Matrix
- Model Comparison Visualization

---

## 📁 Project Structure

```text
loan_project/
│
├── data/
│   └── LoanApprovalPrediction.csv
│
├── models/
│   └── loan_pipeline.pkl
│
├── train.py
├── app.py
│
├── pages/
│   ├── 1_Home.py
│   ├── 2_EDA.py
│   ├── 3_Prediction.py
│   └── 4_Model_Comparison.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run:

```bash
python train.py
```

This will:

- Train the machine learning model
- Evaluate performance
- Save the trained pipeline to:

```text
models/loan_pipeline.pkl
```

---

## 🌐 Run the Streamlit Application

```bash
streamlit run app.py
```

After launching, open:

```text
http://localhost:8501
```

---

## 📈 Dashboard Pages

### 🏠 Home
Project overview and objectives.

### 📊 EDA
Exploratory Data Analysis including:
- Dataset preview
- Missing values
- Correlation heatmap
- Loan status distribution

### 🔮 Prediction
Interactive form for entering applicant information and receiving loan approval predictions.

### 📈 Model Comparison
Performance comparison between multiple machine learning models.

---

## 💻 Example Prediction Workflow

1. Enter applicant information.
2. Specify income and loan details.
3. Submit the application.
4. Receive an approval prediction.

---

## 🛠 Technologies Used

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

## 📚 References

### Streamlit

https://streamlit.io

### Scikit-learn Documentation

https://scikit-learn.org

### Pandas Documentation

https://pandas.pydata.org

### Seaborn Documentation

https://seaborn.pydata.org

---

## 👨‍💻 Author

**Marwan**

Artificial Intelligence & Data Science Graduate

---

## 📄 License

This project is intended for educational and academic purposes.
