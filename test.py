from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Test").getOrCreate()

df = spark.createDataFrame([
    (1, "Alice"),
    (2, "Bob")
], ["id", "name"])

df.show()
