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
X = df_clean[['odometer', 'condition', 'mmr', 'new_body_type', 'model']]
y = df_clean['margin']

# One-hot encode categorical columns (convert strings to numeric)
X = pd.get_dummies(X, columns=['new_body_type', 'model'], drop_first=True)

# Check: confirm all are numeric
print(X.dtypes.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
