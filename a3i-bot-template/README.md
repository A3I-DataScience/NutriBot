This template is based on this repository: https://github.com/arora-r/chatapp-with-voice-and-openai-outline

# To run the  chat bot locally in your python environnement:

1. clone the repo,
2. to run open AI online models you need to expose your open AI project key. Create the key [here](https://platform.openai.com/api-keys), and run in terminal
```
export OPENAI_API_KEY="your_api_key_here"
```
**Warning** there multiple types of key that you can create on open AI platform 

3. run 

```
pip install -r requirements.txt
```

4. run 
```
flask --app server.py --debug run
```


# To run on a docker container:

```
docker build . -t a3i-bot     
docker run -p 8000:8000 a3i-bot
```

But not working because I don't know how to add the Open AI key to the docker image. I suppose I can expose it in the Dockerfile but that is probably not how this is supposed to be done.


# Things you can edit:

You can chose in the `worker.py` to use different tts or stt models. If you chose an alternative model (for instance one to run locally without using any openAI key), you will just need to uncomment the necessary imports and add the relevant libraries to the `requirements.txt`. 

You can also chose which open AI model to refer for tts, stt, and message processing and you can edit the profile of the chat bot in the `server.py`.

Note that the local models will be heavy to install to run and  have a lot of dependencies.
