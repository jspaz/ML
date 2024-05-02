# Databricks notebook source
# DBTITLE 0,--i18n-60a5d18a-6438-4ee3-9097-5145dc31d938
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC # Linear Regression: Improving the Model
# MAGIC
# MAGIC In this notebook we will be adding additional features to our model, as well as discuss how to handle categorical features.
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) Learning Objectives:<br>
# MAGIC
# MAGIC By the end of this lesson, you should be able to;
# MAGIC * Correlation Analysis
# MAGIC * Create a Spark ML Pipeline to fit a model
# MAGIC * Evaluate a model’s performance
# MAGIC * Save and load a model using Spark ML Pipeline

# COMMAND ----------

from pyspark.sql.functions import col, abs, desc

# COMMAND ----------

# DBTITLE 0,--i18n-b44be11f-203c-4ea4-bc3e-20e696cabb0e
# MAGIC %md
# MAGIC ## Load Data from Delta Table

# COMMAND ----------

airbnb_df = spark.sql("""select * from demos.datos.airbnb_sf""")

# COMMAND ----------

# DBTITLE 0,--i18n-f8b3c675-f8ce-4339-865e-9c64f05291a6
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Train/Test Split
# MAGIC
# MAGIC Let's use the same 80/20 split with the same seed as the previous notebook so we can compare our results (unless you changed the cluster config!)

# COMMAND ----------

train_df, test_df = airbnb_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# DBTITLE 0,--i18n-dedd7980-1c27-4f35-9d94-b0f1a1f92839
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC Now we can combine our OHE categorical features with our numeric features.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "price"))]
#numeric_cols2 = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") )]
#assembler_inputs2 = numeric_cols2
assembler_inputs = numeric_cols
#ohe_output_cols +
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

#assembler = VectorAssembler(inputCols=['host_is_superhostOHE', 'cancellation_policyOHE', 'instant_bookableOHE', 'neighbourhood_cleansedOHE', 'property_typeOHE', 'room_typeOHE', 'bed_typeOHE', 'host_total_listings_count', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds',  'minimum_nights', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'price', 'bedrooms_na', 'bathrooms_na', 'beds_na', 'review_scores_rating_na', 'review_scores_accuracy_na', 'review_scores_cleanliness_na', 'review_scores_checkin_na', 'review_scores_communication_na', 'review_scores_location_na', 'review_scores_value_na'], outputCol='features')
#train_df = assembler.transform(train_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Correlation Analysis
# MAGIC

# COMMAND ----------

def correlation_df(df, target_var, feature_cols, method):

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation

    # Assemble features into a vector
    target_var = [target_var]
    feature_cols = feature_cols
    df_cor = df.select(target_var+feature_cols)
    assembler = VectorAssembler(inputCols=target_var+feature_cols, outputCol="features")
    df_cor = assembler.transform(df_cor)

    # Calculate correlation matrix
    correlation_matrix = Correlation.corr(df_cor, "features", method=method).head()

    # Extract the correlation coefficient between target and each feature
    target_corr_list = [correlation_matrix[0][i, 0] for i in range(len(feature_cols)+1)][1:]

    # Create a DataFrame with target variable, feature names, and correlation coefficients
    correlation_data = [(feature_cols[i], float(target_corr_list[i])) for i in range(len(feature_cols))]

    correlation_df = spark.createDataFrame(correlation_data, ["feature", "correlation"])

    correlation_df = correlation_df.withColumn("abs_correlation", abs("correlation"))
    # Print the result
    return correlation_df

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Spearman Correlation

# COMMAND ----------

from pyspark.sql.functions import col, abs, desc
target = 'price'
categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
indep_cols = [x for x in train_df.columns if x not in categorical_cols and x != target]

tr_df = train_df.select([col(field) for (field, dataType) in train_df.dtypes if dataType == "double"])

df = correlation_df(df = tr_df, 
                    target_var = target, 
                    feature_cols = indep_cols,
                    method = 'spearman')
df.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Pearson Correlation

# COMMAND ----------

target = 'price'
categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
indep_cols = [x for x in train_df.columns if x not in categorical_cols and x != target]
columns = [x for x in train_df.columns if x not in categorical_cols ]

tr_df = train_df.select([col(field) for (field, dataType) in train_df.dtypes if dataType == "double"])

df = correlation_df(df = tr_df, 
                    target_var = target, 
                    feature_cols = indep_cols,
                    method = 'pearson')
##########################################################################
correlation_df = df.withColumn("abs_correlation", abs(col("correlation")))
correlation_df.sort(desc("abs_correlation")).show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Correlation Matrix

# COMMAND ----------

from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
import pandas as pd

# Select only numeric columns
columns = ["accommodates",  "bedrooms", "beds", "bathrooms","latitude","number_of_reviews","review_scores_rating"]
data = train_df.select(columns)

# Convert the DataFrame into an RDD of Vectors
rdd_vectors = data.rdd.map(lambda row: Vectors.dense(row))

# Calculate the Pearson correlation matrix using the RDD of Vectors
correlation_matrix = Statistics.corr(rdd_vectors, method="pearson")

correlation_df = pd.DataFrame(correlation_matrix, columns=columns, index=columns)
print("Correlation matrix:")
print(correlation_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Correlation HeatMap

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Convert the correlation matrix to a Pandas DataFrame
correlation_df = pd.DataFrame(correlation_matrix, columns=columns, index=columns)

# Create the heatmap using Seaborn
plt.figure(figsize=(15, 7))
sns.heatmap(correlation_df, annot=True, cmap="coolwarm", cbar_kws={"aspect": 40})
plt.title("Correlation Matrix Heatmap")
plt.show()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

columns = ["accommodates",  "bedrooms", "beds", "bathrooms","latitude","number_of_reviews","review_scores_rating"]

assembler = VectorAssembler(inputCols=columns, outputCol='features')
#train_df = assembler.transform(train_df)

# COMMAND ----------

print(columns)

# COMMAND ----------

#tr_df = train_df.select('accommodates', 'bedrooms', 'beds', 'bathrooms', 'latitude', 'number_of_reviews', 'review_scores_rating',col('features'))

# COMMAND ----------

# DBTITLE 0,--i18n-fb06fb9b-5dac-46df-aff3-ddee6dc88125
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Linear Regression
# MAGIC
# MAGIC Now that we have all of our features, let's build a linear regression model.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="price", featuresCol="features")

# COMMAND ----------

# DBTITLE 0,--i18n-a7aabdd1-b384-45fc-bff2-f385cc7fe4ac
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Pipeline
# MAGIC
# MAGIC Let's put all these stages in a Pipeline. A <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html?highlight=pipeline#pyspark.ml.Pipeline" target="_blank">Pipeline</a> is a way of organizing all of our transformers and estimators.
# MAGIC
# MAGIC This way, we don't have to worry about remembering the same ordering of transformations to apply to our test dataset.

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [assembler, lr]
pipeline = Pipeline(stages=stages)

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# DBTITLE 0,--i18n-c7420125-24be-464f-b609-1bb4e765d4ff
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Saving Models
# MAGIC
# MAGIC We can save our models to persistent storage (e.g. DBFS) in case our cluster goes down so we don't have to recompute our results.

# COMMAND ----------

working_dir = "dbfs:/demos.datos/data/models/temp"

# COMMAND ----------

pipeline_model.write().overwrite().save(working_dir)

# COMMAND ----------

# DBTITLE 0,--i18n-15f4623d-d99a-42d6-bee8-d7c4f79fdecb
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Loading models
# MAGIC
# MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
# MAGIC
# MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic `PipelineModel` back in.

# COMMAND ----------

from pyspark.ml import PipelineModel

saved_pipeline_model = PipelineModel.load(working_dir)

# COMMAND ----------

#pred__df = lr_model.transform(test_df)

# COMMAND ----------

# DBTITLE 0,--i18n-1303ef7d-1a57-4573-8afe-561f7730eb33
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Apply the Model to Test Set

# COMMAND ----------

pred_df = saved_pipeline_model.transform(test_df)

display(pred_df.select("features", "price", "prediction"))

# COMMAND ----------

# DBTITLE 0,--i18n-9497f680-1c61-4bf1-8ab4-e36af502268d
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Evaluate the Model
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) How is our R2 doing?

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")


# COMMAND ----------

# DBTITLE 0,--i18n-cc0618e0-59d9-4a6d-bb90-a7945da1457e
# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC As you can see, our RMSE decreased when compared to the model without one-hot encoding that we trained in the previous notebook, and the R2 increased as well!

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Examine Residuals

# COMMAND ----------

import pyspark.sql.functions as F
predictions_with_residuals = pred_df.withColumn("residual", (F.col("price") - F.col("prediction")))
display(predictions_with_residuals.agg({'residual': 'mean'}))

# COMMAND ----------

import numpy as np 
import statsmodels.api as sm 
import pylab as py 

# Convert DataFrame column to NumPy array
residuals = predictions_with_residuals.select("residual").toPandas().values.flatten()
  
sm.qqplot(residuals, line ='q') 
py.show() 

# COMMAND ----------

import matplotlib.pyplot as plt

plt.scatter(residuals,predictions_with_residuals.select("price").toPandas().values.flatten())
plt.title('Residual vs predicted values')
plt.xlabel("residuals")
plt.ylabel("price")
plt.show()
