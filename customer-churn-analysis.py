from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
data_path = "customer_churn.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Task 1: Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df = df.fillna({'TotalCharges': 0})

    categorical_cols = ['gender', 'PhoneService', 'InternetService']
    indexers = [StringIndexer(inputCol=col, outputCol=col + "Index") for col in categorical_cols]
    encoders = [OneHotEncoder(inputCol=col + "Index", outputCol=col + "Vec") for col in categorical_cols]

    label_indexer = StringIndexer(inputCol="Churn", outputCol="ChurnIndex")
    numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    assembler_inputs = [col + "Vec" for col in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler])
    model = pipeline.fit(df)
    final_df = model.transform(df).select("features", "ChurnIndex")

    return final_df

# Task 2: Splitting Data and Building a Logistic Regression Model
def train_logistic_regression_model(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(labelCol="ChurnIndex", featuresCol="features")
    lr_model = lr.fit(train_df)
    predictions = lr_model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndex", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"Logistic Regression AUC: {round(auc, 2)}")

# Task 3: Feature Selection Using Chi-Square Test
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="ChurnIndex")
    selected = selector.fit(df).transform(df).select("selectedFeatures", "ChurnIndex")
    selected.show(5)

# Task 4: Hyperparameter Tuning with Cross-Validation for Multiple Models
def tune_and_compare_models(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="ChurnIndex", metricName="areaUnderROC")

    def tune(model, param_grid):
        cv = CrossValidator(estimator=model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
        cv_model = cv.fit(train_df)
        best_model = cv_model.bestModel
        auc = evaluator.evaluate(best_model.transform(test_df))
        return auc, best_model

    # Logistic Regression
    print("\nTuning Logistic Regression...")
    lr = LogisticRegression(labelCol="ChurnIndex", featuresCol="features")
    grid_lr = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).addGrid(lr.maxIter, [10, 20]).build()
    auc_lr, best_lr = tune(lr, grid_lr)
    print(f"Best Logistic Regression AUC: {round(auc_lr, 2)} | regParam={best_lr._java_obj.getRegParam()}, maxIter={best_lr._java_obj.getMaxIter()}")

    # Decision Tree
    print("\nTuning Decision Tree...")
    dt = DecisionTreeClassifier(labelCol="ChurnIndex", featuresCol="features")
    grid_dt = ParamGridBuilder().addGrid(dt.maxDepth, [5, 10]).build()
    auc_dt, best_dt = tune(dt, grid_dt)
    print(f"Best Decision Tree AUC: {round(auc_dt, 2)} | maxDepth={best_dt._java_obj.getMaxDepth()}")

    # Random Forest
    print("\nTuning Random Forest...")
    rf = RandomForestClassifier(labelCol="ChurnIndex", featuresCol="features")
    grid_rf = ParamGridBuilder().addGrid(rf.maxDepth, [10, 15]).addGrid(rf.numTrees, [20, 50]).build()
    auc_rf, best_rf = tune(rf, grid_rf)
    print(f"Best Random Forest AUC: {round(auc_rf, 2)} | maxDepth={best_rf._java_obj.getMaxDepth()}, numTrees={best_rf._java_obj.getNumTrees()}")

    # GBT
    print("\nTuning GBT...")
    gbt = GBTClassifier(labelCol="ChurnIndex", featuresCol="features")
    grid_gbt = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]).addGrid(gbt.maxIter, [10, 20]).build()
    auc_gbt, best_gbt = tune(gbt, grid_gbt)
    print(f"Best GBT AUC: {round(auc_gbt, 2)} | maxDepth={best_gbt._java_obj.getMaxDepth()}, maxIter={best_gbt._java_obj.getMaxIter()}")

# === Execute tasks ===
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()
