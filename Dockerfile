FROM python:3.9.16-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
CMD ["python3", "predict.py"]

# docker build -t predict .
# первый том указывает на папку хоста, куда сохраняется вышедшая маска
# второй том указывает на папку хоста, откуда берется изображение для прохождения в сетку
# docker run -d -v path/to/host/predicted_dir:/app/predicted -v path/to/host/image_dir:/app/images <docker image id>