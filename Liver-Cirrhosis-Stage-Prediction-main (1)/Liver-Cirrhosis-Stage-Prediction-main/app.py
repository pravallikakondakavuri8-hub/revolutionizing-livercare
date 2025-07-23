from flask import Flask, render_template, request
import joblib

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.bin')

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ----- Map categorical inputs -----
        drug = 0.0 if request.form['drug'] == "D-penicillamine" else 1.0
        sex = 0.0 if request.form['sex'].lower() == "female" else 1.0
        ascites = 1.0 if request.form['ascites'].upper() == "YES" else 0.0
        hepatomegaly = 1.0 if request.form['hepatomegaly'].upper() == "YES" else 0.0
        spider = 1.0 if request.form['spiders'].upper() == "YES" else 0.0

        # Edema Mapping
        edema_map = {
            "No edema and no diuretic therapy for edema": 0.0,
            "Edema present without diuretics, or edema resolved by diuretics": -1.0,
            "Edema despite diuretic therapy": 1.0
        }
        edema = edema_map.get(request.form['edema'], -1.0)  # Default fallback

        # ----- Numerical inputs -----
        features = [
            drug,
            float(request.form['age']),
            sex,
            ascites,
            hepatomegaly,
            spider,
            edema,
            float(request.form['bilirubin']),
            float(request.form['cholesterol']),
            float(request.form['albumin']),
            float(request.form['copper']),
            float(request.form['alk_phos']),
            float(request.form['sgot']),
            float(request.form['tryglycerides']),
            float(request.form['platelets']),
            float(request.form['prothrombin'])
        ]

        # ----- Prediction -----
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)[0]

        # ----- Interpret Prediction -----
        prediction_map = {
            1: "The person has a normal liver.",
            2: "The person has a fatty liver.",
            3: "The person is suffering from Liver Fibrosis.",
            4: "The person is suffering from Liver Cirrhosis."
        }

        result = prediction_map.get(prediction, "Unknown result.")
        return render_template('home.html', prediction_text=result)

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error occurred: {str(e)}")

# Run the Flask app
if __name__ == "__main__":
    app.run(port=8080, debug=True)
