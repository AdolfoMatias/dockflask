FROM tiangolo/uwsgi-nginx-flask:python3.8

WORKDIR /app/

COPY ./requirements.txt  ./main.py  /app/
RUN pip install --upgrade pip && pip install -r ./requirements.txt


EXPOSE 5000


CMD ["python", "main.py"]