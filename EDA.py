import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis")

data = pd.read_csv("data/LoanApprovalPrediction.csv")
data.drop("Loan_ID", axis=1, inplace=True)

st.write(data.head())

st.subheader("Missing Values")
st.write(data.isnull().sum())

st.subheader("Loan Status Distribution")

fig, ax = plt.subplots()
sns.countplot(x="Loan_Status", data=data, ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)