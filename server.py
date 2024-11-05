import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker  # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)
documents = []
user_informations = {}
first_message = True

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')  # Render the index.html template


# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():

    global user_informations, documents,first_message

    user_informations_old = user_informations.copy()

    user_informations['gender'] = request.json['gender']
    user_informations['age'] = request.json['age']
    user_informations['size'] = request.json['size']
    user_informations['weight'] = request.json['weight']
    user_informations['country'] = request.json['country']

    print('user_informations', user_informations)

    if user_informations!=user_informations_old:


        documents = worker.process_local_documents(country = user_informations['country'])
        worker.process_document(documents, user_informations)

        first_message = False

    user_message = request.json['userMessage']  # Extract the user's message from the request
    print('user_message', user_message)

    bot_response = worker.process_prompt(user_message)  # Process the user's message using the worker module

    # Return the bot's response as JSON
    return jsonify({
        "botResponse": bot_response
    }), 200


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
