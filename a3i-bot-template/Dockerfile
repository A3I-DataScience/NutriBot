FROM python:3.10

WORKDIR /app
COPY . /app


RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg
#RUN pip install git+https://github.com/openai/whisper.git
#RUN pip install git+https://github.com/huggingface/parler-tts.git
RUN pip install -r requirements.txt


# -----------------------------------------------------------------
# Copy certificates to make use of free open ai usage within the lab
# REMOVE THIS WHEN DEPLOYING TO CODE ENGINE

# Copy the self-signed root CA certificate into the container
#COPY certs/AppleRootCA.cer /usr/local/share/ca-certificates/rootCA.cer

# Update the CA trust store to trust the self-signed certificate
#RUN chmod 644 /usr/local/share/ca-certificates/rootCA.crt && \
#  update-ca-certificates

# Hace el puerto 80 disponible para el mundo exterior al contenedor
EXPOSE 80


# Set the environment variable OPENAI_API_KEY to empty string

#ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.cer
#ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.cer
# -----------------------------------------------------------------

CMD ["python",  "server.py"]