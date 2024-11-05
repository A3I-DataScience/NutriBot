import base64
import json
from flask import Flask, render_template, request
from worker import speech_to_text, text_to_speech, openai_process_message
from flask_cors import CORS
import os

import bs4
from langchain_community.document_loaders import WebBaseLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os





from langchain_openai import ChatOpenAI


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

conversation_history = []


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("processing speech-to-text")
    audio_binary = request.data # Get the user's speech from their request
    text = speech_to_text(audio_binary) # Call speech_to_text function to transcribe the speech
    # Return the response back to the user in JSON format
    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    #print(response)
    #print(response.data)
    return response


@app.route('/process-message', methods=['POST'])
def process_message_route():

    bs4_strainer = bs4.SoupStrainer()
    loader = WebBaseLoader(
        web_paths=("https://hal.science/tel-01110542/document",),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
    all_splits = text_splitter.split_documents(docs)

    os.environ["OPENAI_API_KEY"] = "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"

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    llm = ChatOpenAI(model="gpt-4o-mini")

    if len(conversation_history) == 0:

        prompt  = "You can answer general questions about anything the way you like."
        conversation_history.append({"role" : "system", 
                                     "content" : prompt})

    user_message = request.json['userMessage'] # Get user's message from their request
    #print('user_message', user_message)
    conversation_history.append({"role" : "user", "content" : user_message})
    voice = request.json['voice'] # Get user's preferred voice from their request
    #print('voice', voice)
    # Call openai_process_message function to process the user's message and get a response back
    openai_response_text = openai_process_message(conversation_history)
    conversation_history.append({"role" : "assistant", "content" : openai_response_text})
    # Clean the response to remove any emptylines
    openai_response_text = os.linesep.join([s for s in openai_response_text.splitlines() if s])
    # Call our text_to_speech function to convert OpenAI Api's reponse to speech
    openai_response_speech = text_to_speech(openai_response_text, voice)
    # convert openai_response_speech to base64 string so it can be sent back in the JSON response
    openai_response_speech = base64.b64encode(openai_response_speech).decode('utf-8')
    # Send a JSON response back to the user containing their message's response both in text and speech formats
    response = app.response_class(
        response=json.dumps({"openaiResponseText": openai_response_text, "openaiResponseSpeech": openai_response_speech}),
        status=200,
        mimetype='application/json'
    )
    #print(response)
    return response


if __name__ == "__main__":
    app.run(debug=True,  host='0.0.0.0')
