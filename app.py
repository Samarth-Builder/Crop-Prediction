import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Check whether the API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Prediction API is running"})


# Predict the crop
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        data["N"],
        data["P"],
        data["K"],
        data["temperature"],
        data["humidity"],
        data["ph"],
        data["rainfall"]
    ]]

    prediction = model.predict(features)[0]

    return jsonify({"predicted_crop": str(prediction)})


if __name__ == "__main__":
    app.run(debug=True)