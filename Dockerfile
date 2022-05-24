# puxando iamgem de camada existente
FROM tiangolo/uwsgi-nginx-flask:python3.8

# criando diretorio do projeto
WORKDIR /app/

# copiando arquivos necessários para dentro o diretório do rpojeto 
COPY  .flaskenv ./model.pkl ./requirements.txt /app/
COPY  static /app/static 
COPY  templates /app/templates

#rodando os arquivos necessários 
RUN pip install -r ./requirements.txt

#copaidno o arquivo central
COPY main.py  /app/

#expondo a porta que o projeto roda na máquina local
EXPOSE 5000/tcp

#rodando a ferramenta e o arquivo princiapl
CMD ["python", "main.py"]