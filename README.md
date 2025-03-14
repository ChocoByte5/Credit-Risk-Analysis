# Credit-Risk-Analysis
Machine Learning model for predicting loan approval based on financial data.

# Project Overview
This project aims to predict loan approval based on financial data using Machine Learning. It utilizes **Decision Trees** and **hyperparameter tuning** to optimize the accuracy of predictions.

# Features
- Data preprocessing and handling missing values
- Exploratory Data Analysis (EDA)
- Feature engineering and importance analysis
- Decision Tree model with hyperparameter tuning
- Model evaluation and visualization

# Files in this repository
- `credit_risk_analysis_dt_tuning.py` → Python script for training and tuning the Decision Tree model.
- `credit_risk_analysis_DT_Tuning.ipynb` → Jupyter Notebook with code, analysis, and explanations.
- `credit_risk_model.pkl` → Trained Decision Tree model saved for future use.
  
# Results
The final Decision Tree model achieved ***91% accuracy***, balancing performance and generalization.

**Note:** Variable names in the dataset are in Polish, but all explanations, descriptions, and insights are provided in English for clarity.

### Feature Importance in Credit Risk Model
![Feature Importance](credi_risk_features.png)

This bar chart shows the most influential features in determining loan approval. The most significant factor is previous loan defaults, followed by loan percent income and interest rate.

### Training vs Testing Accuracy
![Training vs Testing Accuracy](credi_risk_accuracy.png)

This plot compares the model’s accuracy on training and test datasets. The model achieves around **91% accuracy**, balancing performance and generalization.

### Confusion Matrix for Credit Risk Model
![Confusion Matrix](matrix_credit_risk.png)

The confusion matrix illustrates the model's predictions versus actual outcomes. Most predictions are correct, with a low number of false positives and false negatives.

# Next Steps
Test **Random Forest** or **Gradient Boosting** for improved performance  
Apply **feature selection techniques** to enhance model efficiency 

**Loan Approval Classification Data** (used for credit risk analysis)  
🔗 [Dataset link](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)  

# How to Use the Model
**Install required libraries** (if not already installed): 
```bash
pip install joblib pandas numpy scikit-learn

**Load the trained model:
import joblib
model = joblib.load("credit_risk_model.pkl")
print("Model loaded successfully!")

**Make predictions (example):
import numpy as np
sample_data = np.array([[50000, 12000, 15.0, 5, 700, 1, 30, 1, 0.3, 2]])  
prediction = model.predict(sample_data)
print("Predicted Loan Approval:", prediction)
   

