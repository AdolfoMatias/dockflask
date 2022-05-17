FROM tiangolo/uwsgi-nginx-flask:python3.8

WORKDIR /app/

COPY  .flaskenv ./model.pkl ./requirements.txt /app/
COPY  static /app/static 
COPY  templates /app/templates
RUN pip install -r ./requirements.txt

COPY main.py  /app/

EXPOSE 5000/tcp

CMD ["python", "main.py"]