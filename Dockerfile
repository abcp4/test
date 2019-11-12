FROM tensorflow/tensorflow:latest-gpu-py3
COPY requirements.txt /requirements.txt

RUN apt-get update
RUN apt-get install vim -y
RUN ["apt-get", "install", "-y", "libsm6", "libxext6", "libxrender-dev"]
RUN pip install -r /requirements.txt
RUN mkdir /workspace

EXPOSE 8888
WORKDIR /workspace

ENTRYPOINT ["jupyter","notebook","--ip=0.0.0.0","--allow-root"]
