import datetime
import logging
import os

import pandas as pd
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_file
from flask import url_for
from flask_cors import CORS

import worker  # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)
documents = []
user_informations = {}
first_message = True
meal_plan_df = pd.DataFrame()


# Define the route for the index page
@app.route("/", methods=["GET"])
def index():

    return render_template("index.html")  # Render the index.html template


@app.route("/download")
def download():

    csvs = sorted(
        [file for file in os.listdir() if file.endswith(".csv") and file.startswith("Meal-Plan")]
    )
    if len(csvs) > 0:

        path = csvs[-1]
        return send_file(path, as_attachment=True)

predefined_profiles = {
    "male_1": {
        "gender": "male",
        "age": 25,
        "size": 180,
        "weight": 90,
        "country": "Switzerland"
    },
    "female_1": {
        "gender": "female",
        "age": 55,
        "size": 165,
        "weight": 50,
        "country": "Mexico",
    },
    "male_2": {
        "gender": "male",
        "age": 65,
        "size": 180,
        "weight": 70,
        "country": "China"
    },
    "female_2": {
        "gender": "female",
        "age": 35,
        "size": 170,
        "weight": 50,
        "country": "India",}}

# Define the route for processing messages
@app.route("/process-message", methods=["POST"])
def process_message_route():

    global user_informations, documents, first_message

    user_informations_old = user_informations.copy()

    user_profile = request.json["profile"]

    if user_profile!= 'unknown':
        user_informations = predefined_profiles[user_profile]
    else:

        user_informations["gender"] = request.json["gender"]
        user_informations["age"] = request.json["age"]
        user_informations["size"] = request.json["size"]
        user_informations["weight"] = request.json["weight"]
        user_informations["country"] = request.json["country"]

    print("user_informations", user_informations)

    if user_informations != user_informations_old:

        documents = worker.process_local_documents(country=user_informations["country"])
        worker.process_document(documents, user_informations)

        first_message = False

    user_message = request.json["userMessage"]  # Extract the user's message from the request
    print("user_message", user_message)

    bot_response = worker.process_prompt(
        user_message
    )  # Process the user's message using the worker module

    # Return the bot's response as JSON
    return jsonify({"botResponse": bot_response}), 200


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
