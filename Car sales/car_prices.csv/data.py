## Car Sales Analysis
## This analysis is based on a dataset containing car sales data, including car models, their selling prices, and market prices (MMR).

## This analysis will be used by the sales manager of a car dealership company to understand customer preferences and identify potential profit margins.
## The data contains below columns:

## Step by step as follows:

# Description of the dataset columns

# Year: The manufacturing year of the vehicle (e.g., 2015)
# Make: The brand or manufacturer of the vehicle (e.g., Kia, BMW, Volvo)
# Model: The specific model of the vehicle (e.g., Sorento, 3 Series, S60, 6 Series Gran Coupe)
# Trim: Additional designation for a particular version or option package of the model (e.g., LX, 328i SULEV, T5, 650i)
# Body: The type of vehicle body (e.g., SUV, Sedan)
# Transmission: The type of transmission in the vehicle (e.g., automatic)
# VIN: The Vehicle Identification Number, a unique code used to identify individual motor vehicles
# State: The state in which the vehicle is located or registered (e.g., CA for California)
# Condition: A numerical representation of the condition of the vehicle (e.g., 5.0)
# Odometer: The mileage or distance traveled by the vehicle
# Color: The exterior color of the vehicle
# Interior: The interior color of the vehicle
# Seller: The entity or company selling the vehicle (e.g., Kia Motors America Inc, Financial Services Remarketing)
# MMR: Manheim Market Report, a pricing tool used in the automotive industry
# Selling Price: The price at which the vehicle was sold
# Sale Date: The date and time when the vehicle was sold

# There for the question we can ask:
## Below question will help the sales manager understand customer model preferences and tailor their inventory accordingly.
# 1. Which car model is preferred by customers? 
# 2. Does the preference for car models differ by state?
# 3. Does body type affect the selling count?

## Below question will help the sales manager understand does odometer reading affect the purchasing decision.
# 4. Does odometer reading affect the selling count?

## Does condition is the main factor for the purchasing decision?
# 5. Does condition affect the selling count?

## The main goal of this analysis is to identify potential profit margins.
# 6. Which car has the largest margin difference between MMR and selling price?


# 1. Importing necessary libraries
import pandas as pd

# 2. Load the dataset
df = pd.read_csv("car_prices.csv")

# It is important to create a copy of the original dataset to avoid modifying it directly.
df_original = df.copy()

# 3. The first step is to understand the dataset and its structure.
df.describe()

# Here we can identify the mean, standard deviation, and other statistical measures for numerical columns.
# We can see that the year ranges from 1982 to 2015
# and the odometer readings range from 0 to 999999, which indicates that some vehicles may have been driven extensively or maybe an error in data entry.

# 4. Identify any missing values in the dataset.
df.isnull().sum()

# From the output we can see that the 'transmission' have the highe number of mission which is 65,352 missing values indicate 11.7% of the total dataset.
# I need to understant wether i should remove these rows or remove the column itself.
df['transmission'].value_counts()

# Here the automatic trasmission is the most common, 85.17% of the dataset and after factoring out the null value, the percent rise to  96%. Therefore, the transmission type may not provide significant insights for the analysis.
# Hence, I choose to remove the 'transmission' column from the dataset.
df = df.drop(columns=['transmission'])

# Next, handling the missing values in other columns. I choose to remove the rows with missing column as the number of missing values is relatively low.
df = df.dropna()
# After removing the rows with missing values, it is important to check the shape of the dataset to see how many rows were removed.
df.isnull().sum()
# Now, there are no missing values in the dataset.

# 5. Next, we need to check for any duplicate rows in the dataset.
duplicates = df.duplicated().sum()
print(duplicates)
# No duplicate rows.

# 6. Next, need to check the data types of each column.
df.dtypes
# All column have the right data type except 'sale date' which is object type.
# Therefore, I need to convert the 'sale date' column to datetime type.
df['saledate']= pd.to_datetime(df['saledate'], utc=True)
df.dtypes

# 7. Next, I will check for any outliers in the numerical columns.
import matplotlib.pyplot as plt

numerical_columns = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']

for column in numerical_columns:
    plt.figure(figsize=(10, 5))
    plt.boxplot(df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# The odomoter column shows some extreme outliers.
check = df[df['odometer'] > 800000][['year','saledate','odometer']]
print(check)

# There is 63 values with odometer greater than 800,000km.
# Based on reasearch study The Federal Highway Administration (FHWA) (US since we use the us dataset) states that the average annual mileage for passenger vehicles is around 22.5k km.
# Therefore we need to construct a checking to see wether the data is valid or not.
df['age']= df['saledate'].dt.year - df['year']

# Supprise to my finding, i found new issues where the date of sale is before the manufacturing year. ([age] < 0)
# Considering this is a sales transaction, it is impossible for a car to be sold before it is manufactured.
# Therefore, I need to remove these rows from the dataset.
df = df[df['age'] >= 0]
df['age'].describe()

# Now back to the odometer issue. Instead of using 22.5k km per year, i will use 40k km per year to give some room for high mileage drivers.
df['max_ex_odometer']= df['age']*40000
outliers_odometer = df[df['odometer']> df['max_ex_odometer']]
print(outliers_odometer[['year','saledate','odometer','age','max_ex_odometer']])

# We found a new issues, a false outlier when the age differnce is 0, the odometer will be 0 thus making the odmeter max reading also 0.

# We need to check the month of the date of sale.
check2 = df['saledate'].dt.month.value_counts().sort_index()
print(check2)
check3 = check2.loc[1:5].sum()/check2.sum()
print(check3)
# It shows that 73.9% of the cars are sold in the first half of the year.
# Therefore, for year 0 assuming it use 73.9% of the year to drive, thus the max odometer reading will be 29,650 km rounding off to 30k km.
# Since the finding shows that most cars are sold in the first half of the year, I will set for the age year 1 will be = 30k km (year 0 assumption) + 40k km (the full year assumption as above) = 70k km.
df['max_ex_odometer']= df['age']*70000
df.loc[df['age']==0, 'max_ex_odometer']= 40000
# Check outliers again
outliers_odometer = df[df['odometer']> df['max_ex_odometer']]
print(outliers_odometer[['year','saledate','odometer','age','max_ex_odometer']])

# drop the outliers
df_clean = df[df['odometer']<= df['max_ex_odometer']]
df_clean.describe()

df_clean[df_clean['odometer']>800000]

# From the description, 999k km is still an outlier but it is within the acceptable range based on the max_ex_odometer calculation.
# But to be conservative, I will remove the odometer readings above 800k km.
df_clean = df_clean[df_clean['odometer']<=800000]
df_clean.describe()
# Now the odometer column shows no extreme outliers.

#### We move to the analysis part

# 1. Which car model is preferred by customers, top 5 models?
model_counts = df_clean['model'].value_counts()
model_counts.head(5) 

## The top 5 preferred car models by customers are:
# 1. Altima
# 2. F-150
# 3. Fusion
# 4. Camry
# 5. Escape


# 2. Does the preference for car models differ by state?

state_model_counts = df_clean.groupby('state')['model'].value_counts().unstack().fillna(0)
state_model_counts = state_model_counts[state_model_counts.sum().sort_values(ascending=False).index]
print(state_model_counts)

# From the above table, we can see that the preference for car models does differ by state.

# 3. Does body type affect the selling count?
body_type_unique = df_clean['body'].unique()
print(body_type_unique)

# There are many body type and some of them are refer to the same body type.
# THus the next step is to group them into common body types which are sedan, suv, convertible, coupe, hatchback, wagon, van, minivan and pickup truck)

df_clean['body'] = df_clean['body'].str.lower().str.strip()

new_list = {
# sedans
    'sedan': 'sedan',
    'g sedan': 'sedan',
    # coupes
    'coupe': 'coupe',
    'g coupe': 'coupe',
    'elantra coupe': 'coupe',
    'genesis coupe': 'coupe',
    'cts coupe': 'coupe',
    'g37 coupe': 'coupe',
    'q60 coupe': 'coupe',
    'koup': 'coupe',
    # convertibles
    'convertible': 'convertible',
    'g convertible': 'convertible',
    'g37 convertible': 'convertible',
    'q60 convertible': 'convertible',
    'beetle convertible': 'convertible',
    'granturismo convertible': 'convertible',
    # suv
    'suv': 'suv',
    # hatchback
    'hatchback': 'hatchback',
    # wagon
    'wagon': 'wagon',
    'cts wagon': 'wagon',
    'tsx sport wagon': 'wagon',
    'cts-v wagon': 'wagon',
    # van
    'van': 'van',
    'transit van': 'van',
    'e-series van': 'van',
    'promaster cargo van': 'van',
    'ram van': 'van',
    # minivan
    'minivan': 'minivan',
    # pickup (truck types)
    'crew cab': 'pickup',
    'double cab': 'pickup',
    'access cab': 'pickup',
    'king cab': 'pickup',
    'supercrew': 'pickup',
    'extended cab': 'pickup',
    'supercab': 'pickup',
    'regular cab': 'pickup',
    'quad cab': 'pickup',
    'crewmax cab': 'pickup',
    'mega cab': 'pickup',
    'cab plus': 'pickup',
    'cab plus 4': 'pickup',
    'club cab': 'pickup',
    'regular-cab': 'pickup',
}

df_clean['new_body_type'] = df_clean['body'].replace(new_list)

df_clean['new_body_type'].value_counts()

df_clean['new_body_type'].value_counts().plot(kind='bar', title='Selling Count by Body Type')
plt.show()

# Calculate the percentage of each body type
percentage = df_clean['new_body_type'].value_counts(normalize=True)
percentage

# So, the top 3 body types preferred by customers are:
# 1. Sedan (45%)
# 2. SUV (26%)
# 3. Pickup (8%)

# 4. Does odometer reading affect the selling count?

import seaborn as sns
import matplotlib.pyplot as plt


sns.histplot(df_clean['odometer'], bins=50)
plt.title("Distribution of Cars Sold by Odometer Reading")
plt.xlabel("Odometer (km)")
plt.ylabel("Count of Cars Sold")
plt.show()
# The histogram shows that most cars sold have odometer readings between 0 to 200,000 km.

# Below is the proved to support above finding where people are paying more for lower odometer reading cars.
sns.scatterplot(x='odometer', y='sellingprice', data=df_clean, alpha=0.4)
plt.title("Odometer vs Selling Price")
plt.xlabel("Odometer (km)")
plt.ylabel("Selling Price ($)")
plt.show()

# Yes the odomoter reading does affect the selling count and the lower odometer reading cars are preferred by customers with higher selling price.

# 5. Does condition affect the selling count?

sns.scatterplot(x='condition',y='sellingprice', data=df_clean, alpha=0.4)

sns.histplot(df_clean['condition'], bins=50)
plt.title("Distribution of Cars Sold by Condition")
plt.xlabel("Condition")
plt.ylabel("Count of Cars Sold")
plt.show()

# It seem to have outliner in condition below 10. Othwise the price of car sold perpendicular to the condition rating.

# 6. Which car model has the largest margin difference between MMR and selling price?
df_clean['margin'] = df_clean['mmr'] - df_clean['sellingprice']

# Margin by car model. Top 10.
margin_car_model = df_clean.groupby('make')['margin'].mean().sort_values(ascending=False).head(10)


# To create a ML on margin

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Define features and target
X = df_clean[['odometer', 'condition', 'new_body_type', 'model']]
y = df_clean['sellingprice']

# One-hot encode categorical columns (convert strings to numeric)
X = pd.get_dummies(X, columns=['new_body_type', 'model'], drop_first=True, dtype=float)

# Check: confirm all are numeric
print(X.dtypes.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate model performance
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)


# Print results
print(f"R² on training set: {r2_train:.4f}")
print(f"R² on test set: {r2_test:.4f}")

# The results show R2 training set is 0.9801 while R2 test is 0.8662
# This indicates that the model performs well on unseen data, suggesting that odometer, condition, body type, and model are significant factors in determining the selling price of cars.

# As we know based on the question 2, where the car model differ by state, thus we can try to check R2 by state.

# --- Regional R² analysis ---
states = df_clean['state'].value_counts().index[:5]  # top 5 states by sample count
results = {}

for s in states:
    temp = df_clean[df_clean['state'] == s]
    if len(temp) < 200:  # skip small samples
        continue
    X = temp[['odometer', 'condition', 'new_body_type', 'model']]
    y = temp['sellingprice']
    
    # Encode categorical columns
    X = pd.get_dummies(X, columns=['new_body_type', 'model'], drop_first=True, dtype=float)
    
    # Fill any remaining missing values (optional)
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    model = RandomForestRegressor(random_state=42)  # reinitialize inside loop to avoid state issues
    model.fit(X_train, y_train)
    
    # Evaluate
    r2 = model.score(X_test, y_test)
    results[s] = round(r2, 3)

print("Regional R² results:")
print(results)

# The regional R2 results shows that the model performs consistently well across different states:
# 'fl': 0.863, 'ca': 0.856, 'pa': 0.783, 'tx': 0.794, 'ga': 0.838

# Plotting the regional R² results
results = {'fl': 0.863, 'ca': 0.856, 'pa': 0.783, 'tx': 0.794, 'ga': 0.838}

# Prepare data
states = list(results.keys())
r2_scores = list(results.values())

# Plot
plt.figure(figsize=(8,5))
plt.bar(states, r2_scores, color='skyblue')
plt.ylim(0,1)
plt.title("Regional R² Results for Selling Price Prediction")
plt.xlabel("State")
plt.ylabel("R² Score")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show values on top of bars
for i, v in enumerate(r2_scores):
    plt.text(i, v + 0.01, str(v), ha='center', fontweight='bold')

plt.show()

# --- End of Regional R² analysis ---

# --- Model-specific Linear Regression Equations ---
from sklearn.linear_model import LinearRegression

models = df_clean['model'].value_counts().index  # all models
results = {}

for m in models:
    df_model = df_clean[df_clean['model'] == m]
    
    if len(df_model) < 50:  # skip models with very few samples
        continue
    
    X = df_model[['odometer', 'condition', 'new_body_type']]
    y = df_model['sellingprice']
    
    # One-hot encode body type only for this model
    X = pd.get_dummies(X, columns=['new_body_type'], drop_first=True, dtype=float)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Prepare equation
    intercept = lr.intercept_
    coefs = lr.coef_
    features = X.columns
    equation = f"y = {intercept:.2f} " + " ".join([f"+ {c:.2f}*{f}" for f, c in zip(features, coefs)])
    
    # Store results
    results[m] = {
        'equation': equation,
        'r2_train': lr.score(X_train, y_train),
        'r2_test': lr.score(X_test, y_test)
    }

# Print equation for one model
for model_name, info in results.items():
    print(f"Model: {model_name}")
    print(f"Equation: {info['equation']}")
    print(f"R² Train: {info['r2_train']:.3f}, R² Test: {info['r2_test']:.3f}\n")

# The final finding for the selling price prediction are as below:

# Model: Altima
# Equation: y = 13457.78 + -0.07*odometer + 67.84*condition + -556.01*new_body_type_sedan
# R² Train: 0.754, R² Test: 0.739

# Model: F-150
# Equation: y = 24542.37 + -0.12*odometer + 94.60*condition
# R² Train: 0.558, R² Test: 0.566

# Model: Fusion
# Equation: y = 14775.00 + -0.10*odometer + 65.20*condition
# R² Train: 0.674, R² Test: 0.670

# Model: Camry
# Equation: y = 13066.86 + -0.07*odometer + 76.07*condition + 226.01*new_body_type_sedan + -138.74*new_body_type_wagon
# R² Train: 0.808, R² Test: 0.816

# Model: Escape
# Equation: y = 17880.19 + -0.11*odometer + 60.18*condition
# R² Train: 0.606, R² Test: 0.739

# Model: Focus
# Equation: y = 9476.78 + -0.07*odometer + 54.25*condition + 1994.77*new_body_type_hatchback + 1280.67*new_body_type_sedan + 407.63*new_body_type_wagon
# R² Train: 0.704, R² Test: 0.703

# Model: Accord
# ...
# Model: Astra
# Equation: y = 6160.32 + -0.04*odometer + 36.86*condition
# R² Train: 0.531, R² Test: 0.586

# Now that all the data show good result. We can save our cleaned data in csv for future use.
df_clean.to_csv('df_clean_new.csv', index=False)
