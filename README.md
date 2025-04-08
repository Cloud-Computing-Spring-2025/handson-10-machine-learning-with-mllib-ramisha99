
**Customer Churn Prediction with MLlib (PySpark)**

---

# 📊 Customer Churn Prediction with Apache Spark MLlib

This project uses **Apache Spark MLlib** to predict customer churn based on structured customer data. The model pipeline includes:

- Data preprocessing
- Feature engineering
- Training multiple machine learning models
- Feature selection
- Hyperparameter tuning with cross-validation

---

## Dataset

**File**: `customer_churn.csv`

**Features:**
- `gender`, `SeniorCitizen`, `tenure`, `PhoneService`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `Churn` (target)

---

## ⚙️ Prerequisites

- Apache Spark installed
- Python 3.x
- `pyspark` installed:  
  ```bash
  pip install pyspark
  ```

- Dataset file `customer_churn.csv` placed in your project directory.

---

## How to Run the Project

### Using `python3`
```bash
python3 customer-churn-analysis.py > output.txt
```
### The output file is in the output.txt

##  Task Breakdown and Explanations

---

###  **Task 1: Data Preprocessing and Feature Engineering**

**Goal**: Clean the dataset and convert all features into a format ML models can consume.

 **Steps in Code**:
- Fill missing values in `TotalCharges` with 0
- Convert categorical columns to numerical using `StringIndexer` and `OneHotEncoder`
- Assemble final features using `VectorAssembler`

 **Relevant Code** (in `preprocess_data()`):

```python
df = df.fillna({'TotalCharges': 0})
StringIndexer(inputCol="gender", outputCol="genderIndex")
OneHotEncoder(inputCol="genderIndex", outputCol="genderVec")
VectorAssembler([...], outputCol="features")
```

🖨️ **Sample Output**:
```text
+--------------------+-----------+
|features            |ChurnIndex |
+--------------------+-----------+
|[0.0,12.0,29.85,29...|0.0        |
|[1.0,5.0,53.85,108...|0.0        |
```

---

### Task 2: Train and Evaluate Logistic Regression Model**

**Goal**: Use Logistic Regression to predict churn and evaluate the model's performance.

**Steps in Code**:
- Split data into 80/20 training and testing sets
- Train Logistic Regression model
- Evaluate using `BinaryClassificationEvaluator` with AUC metric

📎 **Relevant Code** (in `train_logistic_regression_model()`):

```python
lr = LogisticRegression(labelCol="ChurnIndex", featuresCol="features")
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
```

🖨️ **Sample Output**:
```text
Logistic Regression AUC: 0.83
```

---

### **Task 3: Feature Selection using Chi-Square Test**

**Goal**: Select the top 5 most relevant features using the Chi-Square test.

**Steps in Code**:
- Use `ChiSqSelector` to choose top 5 features

📎 **Relevant Code** (in `feature_selection()`):

```python
ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures")
```

🖨️ **Sample Output**:
```text
+--------------------+-----------+
|selectedFeatures    |ChurnIndex |
+--------------------+-----------+
|[0.0,29.85,0.0,1.0...|0.0        |
|[1.0,56.95,1.0,0.0...|1.0        |
```

---

###  **Task 4: Hyperparameter Tuning and Model Comparison**

 **Goal**: Compare multiple classification models using 5-fold cross-validation and AUC.

 **Models Used**:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient-Boosted Trees (GBT)

 **Steps in Code** (in `tune_and_compare_models()`):
- Define parameter grids using `ParamGridBuilder`
- Perform cross-validation using `CrossValidator`
- Evaluate with AUC and print best params

🖨️ **Sample Output**:
```text
Tuning Logistic Regression...
Best Logistic Regression AUC: 0.84 | regParam=0.01, maxIter=20

Tuning Decision Tree...
Best Decision Tree AUC: 0.77 | maxDepth=10

Tuning Random Forest...
Best Random Forest AUC: 0.86 | maxDepth=15, numTrees=50

Tuning GBT...
Best GBT AUC: 0.88 | maxDepth=10, maxIter=20
```

---


##  Notes

- Always check for invisible characters in filenames (e.g., `churn_prediction.py​`).
- Redirect output to a file using:
  ```bash
  python3 customer-churn-analysis.py > output.txt
  ```

---
