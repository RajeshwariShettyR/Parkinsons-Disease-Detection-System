from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask app
application = Flask(__name__)
app = application

# Load the trained model and scaler
model = pickle.load(open("parkinsons_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for prediction form
@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/symptoms', methods=['GET'])
def symptoms():
    return render_template('symptoms.html')

# Route for prediction logic
@app.route('/predict', methods=['POST'])
def predict_datapoint():
    result = ""  # Initialize result
    
    try:
        # Retrieve input data from form
        MDVP_Fo_Hz = float(request.form.get("MDVP_Fo_Hz"))
        MDVP_Fhi_Hz = float(request.form.get("MDVP_Fhi_Hz"))
        MDVP_Flo_Hz = float(request.form.get("MDVP_Flo_Hz"))
        MDVP_Jitter_percent = float(request.form.get("MDVP_Jitter_percent"))
        MDVP_Jitter_Abs = float(request.form.get("MDVP_Jitter_Abs"))
        MDVP_RAP = float(request.form.get("MDVP_RAP"))
        MDVP_PPQ = float(request.form.get("MDVP_PPQ"))
        Jitter_DDP = float(request.form.get("Jitter_DDP"))
        MDVP_Shimmer = float(request.form.get("MDVP_Shimmer"))
        MDVP_Shimmer_dB = float(request.form.get("MDVP_Shimmer_dB"))
        Shimmer_APQ3 = float(request.form.get("Shimmer_APQ3"))
        Shimmer_APQ5 = float(request.form.get("Shimmer_APQ5"))
        MDVP_APQ = float(request.form.get("MDVP_APQ"))
        Shimmer_DDA = float(request.form.get("Shimmer_DDA"))
        NHR = float(request.form.get("NHR"))
        HNR = float(request.form.get("HNR"))
        RPDE = float(request.form.get("RPDE"))
        DFA = float(request.form.get("DFA"))
        spread1 = float(request.form.get("spread1"))
        spread2 = float(request.form.get("spread2"))
        D2 = float(request.form.get("D2"))
        PPE = float(request.form.get("PPE"))

        # Create input array
        predict_data = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs,
                                  MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3,
                                  Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2,
                                  D2, PPE]])

        # Standardize input data
        predict_data_scaled = scaler.transform(predict_data)

        # Make prediction
        prediction = model.predict(predict_data_scaled)

        # Determine result
        if prediction[0] == 1:
            result = "The Person has Parkinson's Disease"
            result_color = "text-red-600" 
            result_image = "/static/images/positive.jpg" 
        else:
            result = "The Person does not have Parkinson's Disease"
            result_color = "text-green-600" 
            result_image = "/static/images/negative.jpg" 
    except Exception as e:
        # Handle errors and show them in the response
        result = f"Error processing the input: {e}"

    # Render result on the template
    return render_template('result.html', result=result ,result_color=result_color, result_image=result_image)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
