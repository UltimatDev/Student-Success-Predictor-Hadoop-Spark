import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from src.recommend import generate_recommendation

# -------------------------------
# LOAD SPARK + MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_resources():
    spark = SparkSession.builder \
        .appName("StudentApp") \
        .master("local[*]") \
        .getOrCreate()

    model = PipelineModel.load("hdfs://localhost:9000/user/user/outputs/rf_model")

    return spark, model

spark, model = load_resources()

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Student Predictor", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'>🎓 Student Performance Predictor</h1>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------
#  INPUT SECTION
# -------------------------------
st.subheader("📊 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    study = st.slider("Study Hours / Week", 0, 20, 5)
    attendance = st.slider("Attendance (%)", 0, 100, 75)

with col2:
    score = st.slider("Past Exam Score", 0, 100, 50)
    internet = st.selectbox("Internet Access", ["Yes", "No"])

gender = st.selectbox("Gender", ["Male", "Female"])

parent_edu = st.selectbox(
    "Parental Education",
    ["High School", "Bachelor", "Master"]
)

extra = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

st.divider()

# -------------------------------
#  PREDICTION
# -------------------------------
if st.button("Analyze Student"):

    # Create dataframe matching training schema
    data = [(
        gender,
        study,
        attendance,
        score,
        parent_edu,
        internet,
        extra
    )]

    columns = [
        "Gender",
        "Study_Hours_per_Week",
        "Attendance_Rate",
        "Past_Exam_Scores",
        "Parental_Education_Level",
        "Internet_Access_at_Home",
        "Extracurricular_Activities"
    ]

    df = spark.createDataFrame(data, columns)

    # Predict
    result = model.transform(df)

    prediction = result.select("prediction").collect()[0][0]
    probability = result.select("probability").collect()[0][0]

    # -------------------------------
    #  CONFIDENCE (FIXED LOGIC)
    # -------------------------------
    # probability[1] = Pass probability
    # probability[0] = Fail probability
    pass_prob = float(probability[1])
    fail_prob = float(probability[0])

    confidence = pass_prob if prediction == 1 else fail_prob

    st.divider()

    # -------------------------------
    #  RESULT DISPLAY
    # -------------------------------
    if prediction == 1:
        st.success("Student is Likely to PASS")
        st.metric("Pass Probability", f"{pass_prob:.2f}")
    else:
        st.error("Student is AT RISK")
        st.metric("Risk Probability", f"{fail_prob:.2f}")

    # Confidence bar
    st.progress(confidence)
    st.caption(f"Model Confidence: {confidence:.2f}")

    # -------------------------------
    #  RECOMMENDATIONS
    # -------------------------------
    st.subheader(" Recommendations")

    student = {
        "Study_Hours_per_Week": study,
        "Attendance_Rate": attendance,
        "Past_Exam_Scores": score,
        "Internet_Access_at_Home": internet,
        "Extracurricular_Activities": extra
    }

    recs = generate_recommendation(student)

    for r in recs:
        st.write("•", r)
        
hdfs_df = spark.read.csv(
    "hdfs://localhost:9000/student-data/output_predictions"
)

st.subheader("Recent Predictions (from HDFS)")
st.dataframe(hdfs_df.limit(5).toPandas())
