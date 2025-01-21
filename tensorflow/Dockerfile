FROM tensorflow/tensorflow:latest
WORKDIR /app
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --ignore-installed
EXPOSE 5000
COPY . .
CMD ["flask", "run", "--debug"]