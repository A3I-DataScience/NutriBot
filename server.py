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

import parameters as params

import worker  # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)
documents = []
user_informations = {}
first_message = True

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
        #for file in csvs[:-1]:
        #    os.remove(file)
        return send_file(path, as_attachment=True)

# Define the route for processing messages
@app.route("/process-message", methods=["POST"])
def process_message_route():

    global user_informations, documents, first_message

    user_informations_old = user_informations.copy()

    user_profile = request.json["profile"]

    if user_profile!= 'unknown':
        user_informations = params.predefined_profiles[user_profile]
    else:

        user_informations["gender"] = request.json["gender"]
        user_informations["age"] = request.json["age"]
        user_informations["size"] = request.json["size"]
        user_informations["weight"] = request.json["weight"]
        user_informations["country"] = request.json["country"]

    print("user_informations", user_informations)

    if user_informations != user_informations_old:

        worker.reset_chat_history()
        worker.process_new_profile(user_informations)

        first_message = False

    user_message = request.json["userMessage"]  # Extract the user's message from the request
    print("user_message", user_message)

    bot_response = worker.process_prompt(
        user_message,first_message
    )  # Process the user's message using the worker module
    first_message = False
    # Return the bot's response as JSON
    return jsonify({"botResponse": bot_response}), 200


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
