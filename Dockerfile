FROM tiangolo/uwsgi-nginx-flask:python3.8

WORKDIR /app/

COPY ./requirements.txt Procfile .flaskenv templates/ static/ model.pkl mammographic_masses.data main.py /app/
RUN pip install --upgrade pip && pip install -r ./requirements.txt




EXPOSE 7000


CMD ["python", "main.py"]