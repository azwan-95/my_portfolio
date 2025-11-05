# Vehicle Insurance Fraud Detection
# Vehicle insurance fraud involves conspiring to make false or exaggerated claims involving property damage or personal injuries following an accident. Some common examples include staged accidents where fraudsters deliberately “arrange” for accidents to occur; the use of phantom passengers where people who were not even at the scene of the accident claim to have suffered grievous injury, and make false personal injury claims where personal injuries are grossly exaggerated.

# About this dataset
# This dataset contains vehicle dataset - attribute, model, accident details, etc along with policy details - policy type, tenure etc. The target is to detect if a claim application is fraudulent or not - FraudFound_P

#https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection/data


import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('fraud_oracle.csv')

# Always good to have a backup of the original data
df_backup = df.copy()

# Understanding the data
# As the provider did not provide any data dictionary, we will have to explore the data to understand the features.
# 1. Identifying the columns
print(df.columns)

# 2. Create the base dictionary structure

data_dict = {col: "" for col in df.columns}

data_dict = {
 'Month': 'Accident month',
 'WeekOfMonth': 'Week index within accident month',
 'DayOfWeek': 'Day of week accident occurred',
 'Make': 'Vehicle manufacturer',
 'AccidentArea': 'Accident location type (Urban/Rural)',
 'DayOfWeekClaimed': 'Claim submission day of week',
 'MonthClaimed': 'Claim submission month',
 'WeekOfMonthClaimed': 'Week index within claim month',
 'Sex': 'Policyholder gender',
 'MaritalStatus': 'Policyholder marital status',
 'Age': 'Policyholder age group',
 'Fault': 'Fault attribution (Policy Holder/Third Party)',
 'PolicyType': 'Insurance policy type',
 'VehicleCategory': 'Vehicle category (e.g. Sedan, Sports)',
 'VehiclePrice': 'Vehicle price range',
 'FraudFound_P': 'Fraud flag (1 = fraud, 0 = no fraud)',
 'PolicyNumber': 'Unique policy identifier',
 'RepNumber': 'Claim representative ID',
 'Deductible': 'Claim deductible amount',
 'DriverRating': 'Driver performance rating',
 'Days_Policy_Accident': 'Days from policy start to accident',
 'Days_Policy_Claim': 'Days from policy start to claim',
 'PastNumberOfClaims': 'Count of prior claims by holder',
 'AgeOfVehicle': 'Vehicle age group',
 'AgeOfPolicyHolder': 'Policyholder age range at claim time',
 'PoliceReportFiled': 'Police report status (Yes/No)',
 'WitnessPresent': 'Witness presence status (Yes/No)',
 'AgentType': 'Agent classification (Internal/External)',
 'NumberOfSuppliments': 'Number of supplementary claims',
 'AddressChange_Claim': 'Address change before claim (Yes/No)',
 'NumberOfCars': 'Number of cars insured',
 'Year': 'Accident/claim year',
 'BasePolicy': 'Base policy category (Liability/Collision/All Perils)'
}

# 3. Save the data dictionary to a JSON file for future reference
with open("data_dictionary.json", "w") as f:
    json.dump(data_dict, f, indent=4)

# END of data dictionary creation

## Project Overview
# Stakeholder: Insurance Company Fraud Detection Team
# Purpose: To understand and detect fraudulent vehicle insurance claims
# Audience: Data Scientists, Fraud Analysts, Insurance Underwriters

## Objectives
# To identify which attributes contribute most to fraud detection in insurance claims. 
# Since the dataset has been preprocessed and labeled (FraudFound_P), the goal is to train a supervised machine learning model to determine the most influential features driving fraud outcomes.

## Analysis Goals:
# 1. Build a classification model (e.g. Logistic Regression, Random Forest, XGBoost) using FraudFound_P as the target variable.
# 2. Measure model performance using metrics such as accuracy, precision, recall, and AUC.
# 3. Perform feature importance analysis to identify the top predictors of fraudulent claims.
# 4. Translate results into actionable insights to improve fraud detection policies and risk assessment.

## Questions to Explore
# Which attributes are most predictive of fraudulent insurance claims?


#### Start of ETL Process ####

# 1. Identify the structure of the data
df.shape
print(df.shape)

df.info()

# 2. Check for missing values
df.isnull().sum()
# No missing values found in the dataset

df.duplicated().sum()
# No duplicate records found in the dataset

# 3. Check for numerical descriptive statistics
df.describe()

# chart to vizualise the outliners

df['AgeOfVehicle'].value_counts().plot(kind='bar')
plt.title("Age of Vehicle Distribution")
plt.xlabel("Age of Vehicle")
plt.ylabel("Count")
plt.show()

# 4. Check for categorical descriptive statistics
df.describe(include='object')

## Findings:
# The dataset contains 15,420 insurance claim records.
# Most claims occurred in January and on Mondays.
# The most common vehicle make is Pontiac, and most claimants are male and married.
# Most claims were filed from urban areas and are of the 'Sedan - Collision' policy type.
# The typical policyholder is aged 31–35, with a vehicle around 7 years old.
# Most records show no police report or witness, and claims were handled mainly by external agents.

## Normalising the categorical data
# To ensure the data are readeble and consistent, we will normalise the categorical data.
# From the observation column such as: VehiclePrice, Days_Policy_Accident, Days_Policy_Claim, AgeofVehicle, NumberOfSuppliments, NumberOfCars, AgeOfPolicyHolder appear to be categorical but are represented as numerical data types. We will convert them to categorical data types.


## VehiclePrice ##

df['VehiclePrice'].unique()

# results (['more than 69000', '20000 to 29000', '30000 to 39000', 'less than 20000', '40000 to 59000', '60000 to 69000'])

price_mapping = {
     'less than 20000': 1,
     '20000 to 29000': 2, 
     '30000 to 39000': 3, 
     '40000 to 59000': 4, 
     '60000 to 69000': 5,
     'more than 69000': 6
}

df['VehiclePriceBin']=df['VehiclePrice'].map(price_mapping)

bin_labels = {
     1: 'Below 20k',
     2: '20k–29k', 
     3: '30k–39k', 
     4: '40k–59k', 
     5: '60k–69k',
     6: 'Above 69k'
}

df['VehiclePriceBinLabel'] = df['VehiclePriceBin'].map(bin_labels)

df[['VehiclePrice', 'VehiclePriceBinLabel']].head()


## Days_Policy_Accident ##

df['Days_Policy_Accident'].unique()

# results (['more than 30', '15 to 30', 'none', '1 to 7', '8 to 15'])

DPA_mapping = {
     'none': 1,
     '1 to 7': 2, 
     '8 to 15': 3, 
     '15 to 30': 4, 
     'more than 30': 5,
}

df['Days_Policy_AccidentBin']=df['Days_Policy_Accident'].map(DPA_mapping)

DPA_bin_labels = {
     1: 'none',
     2: '1 to 7', 
     3: '8 to 15', 
     4: '15 to 30', 
     5: 'more than 30',
}

df['Days_Policy_AccidentLabel'] = df['Days_Policy_AccidentBin'].map(DPA_bin_labels)

df[['Days_Policy_Accident', 'Days_Policy_AccidentBin']].head()


## Days_Policy_Claim ##

df['Days_Policy_Claim'].unique()

# results (['more than 30', '15 to 30', '8 to 15', 'none'])

DPC_mapping = {
     'none': 1,
     '1 to 7': 2, 
     '8 to 15': 3, 
     '15 to 30': 4, 
     'more than 30': 5,
}

df['Days_Policy_ClaimBin']=df['Days_Policy_Claim'].map(DPC_mapping)

DPC_bin_labels = {
     1: 'none',
     2: '1 to 7', 
     3: '8 to 15', 
     4: '15 to 30', 
     5: 'more than 30',
}

df['Days_Policy_ClaimLabel'] = df['Days_Policy_ClaimBin'].map(DPC_bin_labels)

df[['Days_Policy_Claim', 'Days_Policy_ClaimBin']].head()

## AgeOfVehicle ##
df['AgeOfVehicle'].unique()
# results (['3 years', '6 years', '7 years', 'more than 7', '5 years', 'new', '4 years', '2 years'])

AOV_mapping = {
        'new': 1,
        '2 years': 2, 
        '3 years': 3, 
        '4 years': 4, 
        '5 years': 5,
        '6 years': 6,
        '7 years': 7,
        'more than 7': 8
    }

df['AgeOfVehicleBin']=df['AgeOfVehicle'].map(AOV_mapping)

AOV_bin_labels = {
        1: 'new',
        2: '2 years', 
        3: '3 years', 
        4: '4 years', 
        5: '5 years',
        6: '6 years',
        7: '7 years',
        8: 'more than 7'
    }
df['AgeOfVehicleLabel'] = df['AgeOfVehicleBin'].map(AOV_bin_labels)
df[['AgeOfVehicle', 'AgeOfVehicleBin']].head()

## NumberOfSuppliments ##
df['NumberOfSuppliments'].unique()
# results (['none', 'more than 5', '3 to 5', '1 to 2'])

NOS_mapping = {
        'none': 1,
        '1 to 2': 2, 
        '3 to 5': 3, 
        'more than 5': 4
    }

df['NumberOfSupplimentsBin']=df['NumberOfSuppliments'].map(NOS_mapping)
NOS_bin_labels = {
        1: 'none',
        2: '1 to 2', 
        3: '3 to 5', 
        4: 'more than 5'
    }
df['NumberOfSupplimentsLabel'] = df['NumberOfSupplimentsBin'].map(NOS_bin_labels)
df[['NumberOfSuppliments', 'NumberOfSupplimentsBin']].head()


## NumberOfCars ##
df['NumberOfCars'].unique()
# results (['3 to 4', '1 vehicle', '2 vehicles', '5 to 8', 'more than 8'])

NOC_mapping = {
        '1 vehicle': 1,
        '2 vehicles': 2, 
        '3 to 4': 3, 
        '5 to 8': 4,
        'more than 8': 5
    }

df['NumberOfCarsBin']=df['NumberOfCars'].map(NOC_mapping)
NOC_bin_labels = {
        1: '1 vehicle',
        2: '2 vehicles', 
        3: '3 to 4', 
        4: '5 to 8',
        5: 'more than 8'
    }
df['NumberOfCarsLabel'] = df['NumberOfCarsBin'].map(NOC_bin_labels)
df[['NumberOfCars', 'NumberOfCarsBin']].head()

## AgeOfPolicyHolder ##
df['AgeOfPolicyHolder'].unique()
# results (['26 to 30', '31 to 35', '41 to 50', '51 to 65', '21 to 25','36 to 40', '16 to 17', 'over 65', '18 to 20'])

AOPH_mapping = {
        '16 to 17': 1,
        '18 to 20': 2, 
        '21 to 25': 3, 
        '26 to 30': 4,
        '31 to 35': 5,
        '36 to 40': 6,
        '41 to 50': 7,
        '51 to 65': 8,
        'over 65': 9
    }

df['AgeOfPolicyHolderBin']=df['AgeOfPolicyHolder'].map(AOPH_mapping)
AOPH_bin_labels = {
        1: '16 to 17',
        2: '18 to 20', 
        3: '21 to 25', 
        4: '26 to 30',
        5: '31 to 35',
        6: '36 to 40',
        7: '41 to 50',
        8: '51 to 65',
        9: 'over 65'
    }
df['AgeOfPolicyHolderLabel'] = df['AgeOfPolicyHolderBin'].map(AOPH_bin_labels)
df[['AgeOfPolicyHolder', 'AgeOfPolicyHolderBin']].head()

# End of normalisation of categorical data

## Save the cleaned and transformed dataset to a new CSV file for further analysis and modeling
df.to_csv('fraud_oracle_cleaned.csv', index=False)

## Vizualisations
# 1. Distribution of Fraudulent vs Non-Fraudulent Claims

ax = df['FraudFound_P'].value_counts().plot(kind='bar', color=['skyblue','salmon'])

for i, v in enumerate(df['FraudFound_P'].value_counts()):
    ax.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

plt.title('Fraud vs Non-Fraud Distribution')
plt.xlabel('FraudFound_P')
plt.ylabel('Count')
plt.show()

percentage_fraud = (df['FraudFound_P'].value_counts()/len(df))*100
print("Percentage Distribution of Fraud vs Non-Fraud Claims:", percentage_fraud)

# 94% of claims are non-fraudulent, while 6% are fraudulent.


