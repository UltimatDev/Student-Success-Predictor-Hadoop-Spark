from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, trim, lower

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# -------------------------------
# 🔥 Start Spark
# -------------------------------
spark = SparkSession.builder.appName("StudentAnalytics").getOrCreate()

# -------------------------------
# 📥 Load from HDFS
# -------------------------------
df = spark.read.csv(
    "hdfs://localhost:9000/student-data/student_performance_dataset.csv",
    header=True,
    inferSchema=True
)

df = df.repartition(4)

print("Number of partitions:", df.rdd.getNumPartitions())

df = df.drop("Student_ID")

# -------------------------------
# 🧼 Label cleaning
# -------------------------------
df = df.withColumn(
    "Pass_Fail_clean",
    lower(trim(col("Pass_Fail")))
)

df = df.withColumn(
    "label",
    when(col("Pass_Fail_clean") == "pass", 1)
    .when(col("Pass_Fail_clean") == "fail", 0)
)

df = df.dropna(subset=["label"])

print("Rows:", df.count())
df.groupBy("label").count().show()

# -------------------------------
# 📊 Spark SQL analytics (HADOOP SHOWCASE)
# -------------------------------
df.createOrReplaceTempView("students")

spark.sql("""
SELECT label, AVG(Attendance_Rate) as avg_attendance
FROM students
GROUP BY label
""").show()

spark.sql("""
SELECT label, AVG(Past_Exam_Scores) as avg_score
FROM students
GROUP BY label
""").show()

# -------------------------------
# FEATURES
# -------------------------------
categorical_cols = [
    "Gender",
    "Parental_Education_Level",
    "Internet_Access_at_Home",
    "Extracurricular_Activities"
]

numeric_cols = [
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores"
]

indexers = [
    StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    for c in categorical_cols
]

encoders = [
    OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_vec")
    for c in categorical_cols
]

assembler = VectorAssembler(
    inputCols=[c + "_vec" for c in categorical_cols] + numeric_cols,
    outputCol="features"
)

rf = RandomForestClassifier(featuresCol="features", labelCol="label")

pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# -------------------------------
# TRAIN
# -------------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train)

# -------------------------------
# PREDICT
# -------------------------------
predictions = model.transform(test)

# -------------------------------
# EVALUATE
# -------------------------------
evaluator = BinaryClassificationEvaluator(labelCol="label")
print("AUC:", evaluator.evaluate(predictions))

predictions.select("label", "prediction", "probability").show(5)

# Force execution
print("Prediction count:", predictions.count())

predictions = predictions.repartition(4)

predictions.select(
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "label",
    "prediction"
).write.mode("overwrite").csv(
    "hdfs://localhost:9000/student-data/output_predictions"
)
# -------------------------------
# SAVE TO HDFS (KEY ADDITION)
# -------------------------------
predictions.select(
    "Study_Hours_per_Week",
    "Attendance_Rate",
    "Past_Exam_Scores",
    "label",
    "prediction"
).write.mode("overwrite").csv(
    "hdfs://localhost:9000/student-data/output_predictions"
)

print("✅ Predictions saved to HDFS")

# -------------------------------
# 💾 SAVE MODEL TO HDFS
# -------------------------------
model.write().overwrite().save(
    "hdfs://localhost:9000/models/rf_model"
)

print("✅ Model saved to HDFS")

spark.stop()
