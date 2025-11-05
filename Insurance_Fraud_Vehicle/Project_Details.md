# Vehicle Insurance Fraud Detection

## 1. Project Overview
Vehicle insurance fraud involves making false or exaggerated claims involving property damage or personal injuries.  
Examples include staged accidents, phantom passengers, and exaggerated injury claims.

This project analyzes a vehicle insurance dataset to detect fraudulent claims using supervised machine learning.

**Stakeholder:** Insurance Company Fraud Detection Team  
**Purpose:** Identify fraudulent vehicle insurance claims  
**Audience:** Data Scientists, Fraud Analysts, Insurance Underwriters  

---

## 2. Dataset Description
**Source:** [Kaggle - Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data)  
**File:** `fraud_oracle.csv`  
**Records:** 15,420  
**Target:** `FraudFound_P` (1 = Fraud, 0 = No Fraud)

**Data Attributes:**  
The dataset includes vehicle, policy, and claim information such as:
- Vehicle Make, Category, and Price  
- Policy Type, Tenure, and Agent Type  
- Accident and Claim details  
- Demographic data of policyholders  

A data dictionary is created and saved as `data_dictionary.json` for reference.

---

## 3. Objectives
1. Identify which attributes contribute most to fraud detection.  
2. Build and evaluate classification models (Logistic Regression, Random Forest).  
3. Handle data imbalance using SMOTE.  
4. Analyze feature importance for insights into fraud behavior.  

---

## 4. ETL Process

**ETL Link:** (https://github.com/azwan-95/my_portfolio/blob/main/Insurance_Fraud_Vehicle/etl.py)  

**Data Checks**
- No missing or duplicate records found.  
- All columns explored for data type and structure.  

**Data Transformation**
Categorical variables with range-based or descriptive text were normalized into numeric bins:
- `VehiclePrice`
- `Days_Policy_Accident`
- `Days_Policy_Claim`
- `AgeOfVehicle`
- `NumberOfSuppliments`
- `NumberOfCars`
- `AgeOfPolicyHolder`

A cleaned version is saved as:  
`fraud_oracle_cleaned.csv`

---

## 5. Data Insights
**General Findings**
- Claims mostly occur in January and on Mondays.  
- Most claimants are male, married, and located in urban areas.  
- Common policy type: *Sedan - Collision*  
- Typical policyholder: 31–35 years old, vehicle age around 7 years.  
- Few reports involve police or witnesses; handled mainly by external agents.

**Fraud Statistics**
- 94% Non-Fraudulent  
- 6% Fraudulent  

---

## 6. Visualizations
- **Fraud Distribution:** Fraud vs Non-Fraud claims  
- **Age Distribution:** Policyholder age breakdown  
- **Fault Attribution:** Policy Holder vs Third Party  

**Key Findings**
- Majority of faults attributed to policyholders (~73%).  
- Fraud cases are rare but show distinct patterns linked to policy and accident timing.

---

## 7. Model Development
**Preprocessing**
- Label encoding for categorical features  
- Standard scaling for numerical variables  
- Class imbalance handled with SMOTE  

**Model Used:** Logistic Regression  
**Train/Test Split:** 80/20  

### Performance Results

| Metric | Class 0 (No Fraud) | Class 1 (Fraud) |
|---------|--------------------|-----------------|
| Precision | 0.99 | 0.13 |
| Recall | 0.60 | 0.89 |
| F1-score | 0.75 | 0.23 |
| Accuracy | 62% | — |

---

## 8. Interpretation
- Model identifies 89% of actual fraud cases (high recall).  
- Precision is low (13%) → many false positives.  
- Indicates a trade-off between detection rate and false alarms.  
- Useful for initial fraud screening, followed by human review.

---

## 9. Next Steps
1. Experiment with advanced algorithms (Random Forest, XGBoost).  
2. Apply feature selection and hyperparameter tuning.  
3. Evaluate cost-sensitive learning to reduce false positives.  
4. Visualize feature importance to guide fraud policy improvements.  

---

## 10. Files in Repository
| File | Description |
|------|--------------|
| `fraud_oracle.csv` | Raw dataset |
| `fraud_oracle_cleaned.csv` | Cleaned dataset |
| `data_dictionary.json` | Column definitions |
| `fraud_detection.ipynb` | Full notebook with code and analysis |
| `README.md` | Project overview (this file) |

---

## 11. Tools and Libraries
- **Python**: pandas, seaborn, matplotlib  
- **Machine Learning**: scikit-learn, imbalanced-learn (SMOTE)  
- **Visualization**: seaborn, matplotlib  

---

## 12. Conclusion
This project demonstrates an end-to-end fraud detection workflow:
- Data exploration  
- Cleaning and normalization  
- Imbalance correction  
- Model building and evaluation  

Initial results highlight a strong fraud capture rate but with a need for better precision.  
Further refinement using ensemble or cost-sensitive models will enhance real-world applicability.

---
