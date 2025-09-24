from flask import Flask, render_template, request
import re
import pandas as pd
import pickle
import tensorflow as tf

app = Flask(__name__)

# === load model and preprocessor ===
model = tf.keras.models.load_model("adaptive_workout_model_rnn.h5")

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("columns.pkl", "rb") as f:
    expected_features = pickle.load(f)

with open("exercises_mapping.pkl", "rb") as f:
    exercises_mapping = pickle.load(f)

with open("diet_mapping.pkl", "rb") as f:
    diet_mapping = pickle.load(f)

with open("equipment_mapping.pkl", "rb") as f:
    equipment_mapping = pickle.load(f)

# --- dictionaries for JPEG icons ---
exercise_images = {
    "squats": "squats.jpeg",
    "deadlifts": "deadlifts.jpeg",
    "bench presses": "bench_press.jpeg",
    "overhead presses": "overhead_press.jpeg",
    "yoga": "yoga.jpeg",
    "walking": "walking.jpeg",
    "brisk walking": "brisk_walking.jpg",
    "cycling": "cycling.jpeg",
    "swimming": "swimming.jpeg",
    "dancing": "dancing.jpeg"
}

equipment_images = {
    "dumbbells": "dumbbells.jpeg",
    "barbells": "barbells.jpeg",
    "resistance bands": "resisitance_bands.jpeg",
    "light athletic shoes": "shoes.jpeg",
    "ellipticals": "elliptical.jpeg",
    "indoor rowers": "rower.jpeg",
    "treadmills": "treadmill.jpeg",
    "rowing machine": "rower.jpeg",
    
}

# --- helper functions ---
def parse_paragraph(text):
    age_match = re.search(r"(\d+)\s*-?\s*year", text, re.I)
    age = int(age_match.group(1)) if age_match else 25

    if re.search(r"\bmale\b", text, re.I):
        sex = "male"
    elif re.search(r"\bfemale\b", text, re.I):
        sex = "female"
    else:
        sex = "male"

    h_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm|m)", text, re.I)
    if h_match:
        val = float(h_match.group(1))
        if 'cm' in text.lower():
            height = val/100
        else:
            height = val
    else:
        height = 1.7

    w_match = re.search(r"(\d+(?:\.\d+)?)\s*kg", text, re.I)
    weight = float(w_match.group(1)) if w_match else 70

    bmi_match = re.search(r"bmi\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.I)
    bmi = float(bmi_match.group(1)) if bmi_match else round(weight/(height**2),2)

    if re.search(r"\bbeginner\b", text, re.I):
        level = "beginner"
    elif re.search(r"\bintermediate\b", text, re.I):
        level = "intermediate"
    elif re.search(r"\badvanced\b", text, re.I):
        level = "advanced"
    else:
        level = "beginner"

    if re.search(r"weight\s*loss", text, re.I):
        goal = "weight_loss"
    elif re.search(r"weight\s*gain", text, re.I):
        goal = "weight_gain"
    elif re.search(r"maintain|maintenance", text, re.I):
        goal = "maintenance"
    else:
        goal = "weight_loss"

    return age, height, weight, bmi, sex, level, goal

def predict_from_paragraph(text):
    age, height, weight, bmi, sex, level, goal = parse_paragraph(text)

    X_new = pd.DataFrame([{
        "Age": age,
        "Height": height,
        "Weight": weight,
        "BMI": bmi,
        "Sex": sex,
        "Level": level,
        "Fitness Goal": goal
    }])

    X_processed = preprocessor.transform(X_new)
    exercises_pred, diet_pred, equipment_pred = model.predict(X_processed)

    exercises_class = exercises_pred.argmax(axis=1)[0]
    diet_class = diet_pred.argmax(axis=1)[0]
    equipment_class = equipment_pred.argmax(axis=1)[0]

    return (
        f"🏋️ Exercises: {exercises_mapping[exercises_class]}",
        f"🥗 Diet: {diet_mapping[diet_class]}",
        f"🛠 Equipment: {equipment_mapping[equipment_class]}"
    )

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    ex_images_list = []
    eq_images_list = []
    if request.method == "POST":
        paragraph = request.form.get("paragraph")
        ex, di, eq = predict_from_paragraph(paragraph)
        result = {"ex": ex, "di": di, "eq": eq}

        text_ex = ex.lower()
        for k, img in exercise_images.items():
            if k in text_ex:
                ex_images_list.append(img)

        text_eq = eq.lower()
        for k, img in equipment_images.items():
            if k in text_eq:
                eq_images_list.append(img)

    return render_template("index.html",
                           result=result,
                           ex_images=ex_images_list,
                           eq_images=eq_images_list)

if __name__ == "__main__":
    app.run(debug=True)
