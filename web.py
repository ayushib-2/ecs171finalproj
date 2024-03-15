from flask import Flask, render_template_string, request
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

app = Flask(__name__)

# Define a simple HTML template
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Learning Model for Stroke Prediction</title>

    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 20px;
        }

        h1 {
            color: #3498db;
        }

        form {
            max-width: 500px;
            margin: auto;
            margin-left: 0;
            display: flex;
            flex-wrap: wrap;
            
        }

        form > div {
            flex: 1 0 calc(50% - 20px);
            margin-bottom: 16px;
            margin-right: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #2c3e50;
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #2c3e50;
        }

        select, input {
            width: 100%;
            padding: 8px; 
            margin-bottom: 16px;
            box-sizing: border-box;
        } 

        input[type="submit"] {
            background-color: #3498db;
            color: white;
            cursor: pointer;
            width: 100%;
        }

        #prediction-result {
            margin-top: 20px;
            color: #27ae60; /* Default: Low risk */
        }

        #prediction-result.high-risk {
            color: #e74c3c; /* High risk color: Red */
        }
    </style>


</head>
<body>
    <h1>Machine Learning Model for Stroke Prediction</h1>
    <form action="/predict" method="post">
        <div>
            <label for="gender">Gender:</label>
            <select id="gender" name="gender">
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
            </select><br>
        </div>

        <div>
            <label for="age">Age:</label>
            <input type="number" id="age" name="age" required><br>
        </div>

        <div>
            <label for="hypertension">Hypertension:</label>
            <select id="hypertension" name="hypertension">
                <option value="Yes">True</option>
                <option value="No">False</option>
            </select><br>
        </div>

        <div>
            <label for="heart_disease">Heart Disease:</label>
            <select id="heart_disease" name="heart_disease">
                <option value="Yes">True</option>
                <option value="No">False</option>
            </select><br>
        </div>

        <div>
            <label for="ever_married">Ever Married:</label>
            <select id="ever_married" name="ever_married">
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select><br>
        </div>

        <div>
            <label for="work_type">Work Type:</label>
            <select id="work_type" name="work_type">
                <option value="children">Children</option>
                <option value="Govt_job">Govt Job</option>
                <option value="Never_worked">Never Worked</option>
                <option value="Private">Private</option>
                <option value="Self-employed">Self-employed</option>
            </select><br>
        </div>

        <div>
            <label for="Residence_type">Residence Type:</label>
            <select id="Residence_type" name="Residence_type">
                <option value="Urban">Urban</option>
                <option value="Rural">Rural</option>
            </select><br>
        </div>

        <div>
            <label for="avg_glucose_level">Avg Glucose Level:</label>
            <input type="number" id="avg_glucose_level" name="avg_glucose_level" step="any" required><br>
        </div>

        <div>
            <label for="bmi">BMI:</label>
            <input type="number" id="bmi" name="bmi" step="any" required><br>
        </div>

        <div>
            <label for="smoking_status">Smoking Status:</label>
            <select id="smoking_status" name="smoking_status">
                <option value="never smoked">Never Smoked</option>
                <option value="formerly smoked">Formerly Smoked</option>
                <option value="smokes">Smokes</option>
            </select><br>
        </div>
        <input type="submit" value="Submit">
    </form>

    <div id="prediction-result" style="color: {{ result_color }};"> </div>

    <script>
        document.addEventListener("DOMContentLoaded", function() {
        document.querySelector("form").addEventListener("submit", function(event) {
        event.preventDefault(); 

        var formData = new FormData(this);

        fetch("/predict", {
                method: "POST",
                body: formData
            })
            .then(response => response.text())
            .then(result => {
                document.getElementById("prediction-result").innerHTML = result;
            })
            .catch(error => console.error('Error:', error));
            });
        });
    </script>
</body>
</html>

"""

# Load the machine learning model
model = joblib.load('random_forest_model.pkl')

# Load LabelEncoders for each category
gender_label_encoder = joblib.load('./encoders/gender_label_encoder.pkl')
ever_married_label_encoder = joblib.load('./encoders/ever_married_label_encoder.pkl')
work_type_label_encoder = joblib.load('./encoders/work_type_label_encoder.pkl')
residence_type_label_encoder = joblib.load('./encoders/residence_type_label_encoder.pkl')
smoking_status_label_encoder = joblib.load('./encoders/smoking_status_label_encoder.pkl')

# Define a function to print encodings
def print_encodings(label_encoder, name):
    print(f"{name} Encodings:")
    for value, encoding in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"{value} -> {encoding}")

# Print encodings for each label encoder
print_encodings(gender_label_encoder, 'Gender')
print_encodings(ever_married_label_encoder, 'Ever Married')
print_encodings(work_type_label_encoder, 'Work Type')
print_encodings(residence_type_label_encoder, 'Residence Type')
print_encodings(smoking_status_label_encoder, 'Smoking Status')

# Load MinMax scalers
age_minmax_scaler = joblib.load('./scalers/age_minmax_scaler.pkl')
avg_glucose_level_minmax_scaler = joblib.load('./scalers/avg_glucose_level_minmax_scaler.pkl')
bmi_minmax_scaler = joblib.load('./scalers/bmi_minmax_scaler.pkl')

def preprocess_input(data):
    columns_to_normalize = ['age', 'avg_glucose_level', 'bmi']
    
    # Scale the numerical values
    data['age'] = age_minmax_scaler.transform(data['age'].values.reshape(-1, 1))[0]
    data['avg_glucose_level'] = avg_glucose_level_minmax_scaler.transform(data['avg_glucose_level'].values.reshape(-1, 1))[0]
    data['bmi'] = bmi_minmax_scaler.transform(data['bmi'].values.reshape(-1, 1))[0]

    # Convert "Yes" and "No" to 1 and 0
    data['heart_disease'] = data['heart_disease'].map({'Yes': 1, 'No': 0})
    data['hypertension'] = data['hypertension'].map({'Yes': 1, 'No': 0})

    # Encode categorical features using respective label encoders
    data['gender'] = gender_label_encoder.transform(data['gender'].values)[0]
    data['ever_married'] = ever_married_label_encoder.transform(data['ever_married'].values)[0]
    data['work_type'] = work_type_label_encoder.transform(data['work_type'].values)[0]
    data['Residence_type'] = residence_type_label_encoder.transform(data['Residence_type'].values)[0]
    data['smoking_status'] = smoking_status_label_encoder.transform(data['smoking_status'].values)[0]

    return data

# Define a route to render the HTML template
@app.route("/")
def home():
    return render_template_string(html_template)

# Define a route to handle form submission and make predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get user input data from the form
    gender = request.form.get("gender")
    age = float(request.form.get("age"))
    hypertension = request.form.get("hypertension")
    heart_disease = request.form.get("heart_disease")
    ever_married = request.form.get("ever_married")
    work_type = request.form.get("work_type")
    residence_type = request.form.get("Residence_type")
    avg_glucose_level = float(request.form.get("avg_glucose_level"))
    bmi = float(request.form.get("bmi"))
    smoking_status = request.form.get("smoking_status")

    # Create a dictionary with user input data
    user_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status
    }

    # Convert the user data to a DataFrame for preprocessing
    user_df = pd.DataFrame([user_data])

    # Preprocess the user input data
    user_df = preprocess_input(user_df)

    # Make a prediction using the loaded model
    prediction = model.predict(user_df)

    # Display the prediction result
    print("Input:")
    print(user_df)
    print(f"Result: {prediction}")
    result = "High Risk of Stroke" if prediction[0] == 1 else "Low Risk of Stroke"
    result_class = "high-risk" if result == "High Risk of Stroke" else ""

    return f'<h2 id="prediction-result" class="{result_class}">Prediction Result: {result}</h2>' 

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)