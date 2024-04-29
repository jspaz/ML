# Databricks notebook source
# Read the table from the given S3 path
df = spark.read.format("delta").load("s3://databricks-workspace-stack-ad9a4-bucket/nvirginia-prod/2244260018898615/user/hive/warehouse/delta_table_test")
display(df)

# COMMAND ----------

# Read the table from the given S3 path
df = spark.read.format("delta").load("s3://databricks-workspace-stack-ad9a4-bucket/nvirginia-prod/2244260018898615/user/hive/warehouse/delta_table_test")

# Create the Unity catalog table
df.write.format("delta").mode("overwrite").saveAsTable("main.default.delta_table_test_migrado")
