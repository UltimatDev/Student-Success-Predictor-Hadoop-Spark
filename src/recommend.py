def generate_recommendation(student):
    recs = []

    if student["Study_Hours_per_Week"] < 5:
        recs.append("Increase study hours")

    if student["Attendance_Rate"] < 70:
        recs.append("Improve attendance consistency")

    if student["Past_Exam_Scores"] < 50:
        recs.append("Revise fundamental concepts")

    if student["Internet_Access_at_Home"] == "No":
        recs.append("Provide offline learning resources")

    if student["Extracurricular_Activities"] == "No":
        recs.append("Engage in extracurricular activities")

    if len(recs) == 0:
        recs.append("Keep up the excellent performance!")

    return recs