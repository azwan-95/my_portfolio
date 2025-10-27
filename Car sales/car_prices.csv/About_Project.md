
# Define the markdown content
## Car Sales Analysis

### Project Overview
This analysis is based on a dataset containing car sales data, including car models, selling prices, and market prices (MMR).  
It aims to help a **car dealership sales manager** understand customer preferences and identify potential profit margins.

---

### Dataset Description
The dataset includes the following columns:

| Column | Description |
|---------|--------------|
| **Year** | Manufacturing year of the vehicle |
| **Make** | Brand or manufacturer (e.g., Kia, BMW, Volvo) |
| **Model** | Specific model (e.g., Sorento, 3 Series, S60) |
| **Trim** | Version or option package (e.g., LX, 328i SULEV) |
| **Body** | Vehicle body type (e.g., SUV, Sedan) |
| **Transmission** | Transmission type (e.g., Automatic) |
| **VIN** | Unique vehicle identification number |
| **State** | Vehicle registration state |
| **Condition** | Numerical rating of vehicle condition |
| **Odometer** | Mileage (distance traveled) |
| **Color** | Exterior color |
| **Interior** | Interior color |
| **Seller** | Seller or company name |
| **MMR** | Manheim Market Report price |
| **Selling Price** | Price at which the vehicle was sold |
| **Sale Date** | Date and time of sale |

---

### Business Questions
1. Which car models are most preferred by customers?  
2. Does model preference vary by state?  
3. Does body type affect sales count?  
4. Does odometer reading influence purchase decisions?  
5. Does condition rating affect sales?  
6. Which cars have the largest profit margin (MMR - Selling Price)?

---

### Data Preparation Steps
1. Imported and explored dataset (`pandas`, `matplotlib`).  
2. Checked missing values and dropped unnecessary columns (e.g., `transmission`).  
3. Removed null values and duplicates.  
4. Converted `sale date` to datetime format.  
5. Handled outliers in `odometer` and `age` logic.  
6. Cleaned unrealistic data (e.g., sale before manufacturing).  
7. Standardized `body type` values into common categories.  

---

### Key Findings
- **Top 5 car models sold:** Altima, F-150, Fusion, Camry, Escape.  
- **Yes, each state have different preference for the car model.
- **Top 3 body types:** Sedan (45%), SUV (26%), Pickup (8%).  
- **Odometer impact:** Cars with lower mileage sell faster and at higher prices.  
- **Condition:** Higher condition scores correlate with higher selling prices.  
- **Profit margin insight:** Certain makes have consistent positive margins, showing potential for high-profit resale.  

---

### Machine Learning Model
A **Random Forest Regressor** was trained to predict the margin (`MMR - Selling Price`) using features:
- Odometer  
- Condition  
- MMR  
- Body type  
- Model  

After one-hot encoding categorical variables, the dataset was split 80/20 for training and testing.  
This helps forecast profitable cars and support data-driven decision-making.

---

### Tools and Libraries
- **Python**: Data analysis and modeling  
- **Pandas**: Data manipulation  
- **Matplotlib / Seaborn**: Visualization  
- **Scikit-learn**: Machine learning  

---

### Summary
This analysis provides insights into customer preferences, vehicle condition impact, and profit margins.  
It demonstrates how data cleaning, exploratory analysis, and predictive modeling can improve decision-making in the automotive sales industry.