import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker  # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    
    # TODO: Extract the user's message from the request
    # TODO: Process the user's message using the worker module
    # TODO: Return the bot's response as JSON
    
# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    # TODO: Check if a file was uploaded  --> use if 'file' not in request file then return jsofify message and error code 400
    # TODO: Save the uploaded file --> extracte the file with request 
    # TODO: Process the saved file --> create proper file_path for saving
    # TODO: Send a response indicating that the document is ready for queries --> retrun jsonify message with code 200

# Run the Flask app (later you need to add "python main block" ==> if __name__ == "__main__":
app.run(debug=True, port=8000, host='0.0.0.0')
