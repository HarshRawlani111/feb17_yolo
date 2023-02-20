FROM python:3.8-slim-buster


WORKDIR /app
#since we are using the COPY command we dont need to ADD files individually
ADD requirements.txt /
ADD com_in_ineuron_ai_utils /
ADD templates /
ADD changing_dir.py /
ADD Dockerfile /
ADD flask_app.py /
ADD inputImage.jpg /
ADD logging_app.py /
ADD predict.py /
ADD repo_cloned.py /
ADD yolo_weights.py /
ADD yolov3.cfg /

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

RUN apt-get update -y 

#RUN pip install -r requirements.txt   
    

#RUN pip install -r requirements.txt

CMD ["python", "flask_app.py"]