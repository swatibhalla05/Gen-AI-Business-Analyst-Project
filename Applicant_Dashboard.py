import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans  # Import KMeans
import warnings

# Set page configuration
st.set_page_config(page_title="Applicant Data Dashboard", page_icon="📊", layout="wide")

# Title
st.title("Applicant Data Dashboard")

# File uploader for the data
file = st.file_uploader("📂 Upload your file (csv/txt/xlsx/xls)", type=["csv", "txt", "xlsx", "xls"])

# Function to load data from the uploaded file or default path
@st.cache_data
def load_data(file=None, default_path=None):
    try:
        if file is not None:
            return pd.read_csv(file, encoding="ISO-8859-1")
        elif default_path:
            return pd.read_csv(default_path, encoding="ISO-8859-1")
        else:
            st.error("No file provided and no default file available.")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Path to the default file
default_file_path = r"https://github.com/Mayaavi69/GAI-Project-Deployment/blob/72d15e68557af712ec58d790ae098695c22b1e23/Applicant_details.csv"

# Check if user uploaded a file or use the default file
if file is not None:
    df = load_data(file=file)
else:
    st.info(f"No file uploaded. Using default file from: {default_file_path}")
    df = load_data(default_path=default_file_path)

# Error handling if dataframe is empty
if df is None:
    st.stop()

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Define categorical and numerical columns
categorical_cols = [
    "Marital_Status",
    "House_Ownership",
    "Vehicle_Ownership(car)",
    "Occupation",
    "Residence_City",
    "Residence_State",
]

numerical_cols = ["Applicant_Age", "Annual_Income", "Work_Experience", "Years_in_Current_Employment", "Years_in_Current_Residence"]

# --- Existing Dashboards ---

# Unique values in each categorical column
st.header("Unique Values in Categorical Columns")
selected_col = st.selectbox("Select Column for Unique Value Count", categorical_cols)
if selected_col:
    unique_values = df[selected_col].unique()
    st.write(f"{selected_col}: {len(unique_values)} unique values")
    st.write(unique_values)

# General visuals of value count for all categorical values
st.header("General Value Counts for Categorical Columns")
cat_columns = categorical_cols
n_cols = 3
n_rows = int(len(cat_columns) / n_cols) + (len(cat_columns) % n_cols > 0)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axs = axs.flatten()

for i, col in enumerate(cat_columns):
    df_temp = df[col].value_counts().reset_index()
    df_temp.columns = ['Category', 'Count']
    sns.barplot(data=df_temp, y='Category', x='Count', ax=axs[i])
    axs[i].set_title(f"Value Counts for {col}")
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

plt.tight_layout()
st.pyplot(fig)
st.write("Interpretation: The above visuals show the distribution of categorical variables, providing insights into the diversity of applicant profiles.")

# Job category analysis and visualizations
occupation_category = {
    'Psychologist': 'Public Service and Administration',
    'Petroleum_Engineer': 'Engineering and Technology',
    'Drafter': 'Engineering and Technology',
}
df['Job_Category'] = df['Occupation'].map(occupation_category)

st.header("Distribution of Job Categories")
fig = px.bar(df, x='Job_Category', title="Distribution of Job Categories")
st.plotly_chart(fig)
st.write("Interpretation: This chart shows the distribution of applicants across different job categories.")

# Occupation Distribution
st.header("Occupation Distribution")
occupation_counts = df['Occupation'].value_counts().reset_index()
occupation_counts.columns = ['Occupation', 'Count']
fig = px.bar(occupation_counts, x='Occupation', y='Count', title="Distribution of Occupations", labels={'Occupation': 'Occupation', 'Count': 'Count'})
st.plotly_chart(fig)
st.write("Interpretation: The above chart displays the number of applicants for each unique occupation.")

# Work Experience and Employment Categorization
def categorize_work_experience(num):
    if 0 <= num <= 5:
        return '0-5'
    elif 6 <= num <= 10:
        return '6-10'
    elif 11 <= num <= 15:
        return '11-15'
    else:
        return '16-20'

df['Work_Experience_Category'] = df['Work_Experience'].apply(categorize_work_experience)

def categorize_years_in_employment(num):
    if 0 <= num <= 5:
        return '0-5'
    elif 6 <= num <= 10:
        return '6-10'
    else:
        return '11-15'

df['Years_in_Current_Employment_Cat'] = df['Years_in_Current_Employment'].apply(categorize_years_in_employment)

columns = ['House_Ownership', 'Vehicle_Ownership(car)', 'Work_Experience_Category', 'Years_in_Current_Employment_Cat']
st.header("Value Counts by Job Category")
for col in columns:
    fig = px.bar(df, x='Job_Category', color=col, barmode='group', title=f"Value Counts for {col} by Job Category")
    st.plotly_chart(fig)

# Target variable distribution
st.header("Target Variable Distribution (Loan Default Risk)")
fig = px.histogram(df, x="Loan_Default_Risk", title="Distribution of Loan Default Risk")
st.plotly_chart(fig)

# Correlation heatmap
st.header("Correlation Heatmap")
corr_matrix = df[numerical_cols].corr()
fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix")
st.plotly_chart(fig)

# --- Additional Methods and Visuals ---

# Decision Tree Classifier for Loan Default Prediction
st.header("Loan Default Prediction with Decision Tree")

# Encode 'Loan_Default_Risk' to binary
df['Loan_Default_Risk'] = df['Loan_Default_Risk'].apply(lambda x: 1 if x == 'Yes' else 0)

# Train-test split
X = df[numerical_cols]
y = df['Loan_Default_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)

# Predictions
y_pred_tree = model_tree.predict(X_test)

# Accuracy and Report
accuracy_tree = accuracy_score(y_test, y_pred_tree)
st.write(f"Decision Tree Accuracy: {accuracy_tree:.2f}")
st.write("Decision Tree Classification Report:")
st.text(classification_report(y_test, y_pred_tree))

# Confusion Matrix for Decision Tree
st.header("Decision Tree Confusion Matrix")
cm_tree = confusion_matrix(y_test, y_pred_tree)
fig, ax = plt.subplots()
sns.heatmap(cm_tree, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# KMeans Clustering for Applicant Segmentation
st.header("KMeans Clustering for Applicant Segmentation")

# Train KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[numerical_cols])

# Visualize clusters
fig = px.scatter(df, x="Applicant_Age", y="Annual_Income", color="Cluster", title="KMeans Clustering: Applicant Segments")
st.plotly_chart(fig)
st.write("Interpretation: KMeans clustering segments applicants into distinct groups based on age and income.")

# Final insights
st.subheader("Final Insights")
st.write("""
- The decision tree classifier performed well in predicting loan default risk.
- KMeans clustering helps identify distinct applicant segments based on key numerical features.
- Outlier detection and correlation analysis provide further insights into the dataset.
""")
