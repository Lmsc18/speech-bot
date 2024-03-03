FROM ubuntu:latest

WORKDIR /speechbot
RUN apt-get update && apt-get install -y python3 python3-pip
RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg
RUN pip3 install -r requirements.txt
COPY . .

CMD ["python3", "main.py"]