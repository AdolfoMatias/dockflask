from flask import Flask, render_template, request
import pickle
import os

#criando o app com flask
app = Flask(__name__)

#criando a página inicial
@app.route("/")
@app.route("/index")
def pagina():
    return render_template("index.html")

#utilizando o metodos post com os formulários do index.html
@app.route("/prever", methods=["POST"])
def predicao():
    #exame biraids, escala de 0 -5
    biraids = request.form.get("biraids")
    #idade da paciente
    age =request.form.get("age")

    #formato da massa do cisto, escala de 0 - 5
    shapemass = request.form.get("shapemass")

    #margem da massa do cisto, escala de 0 - 5
    margemass = request.form.get("margemass")

    #densidade do cist, escala de 0 - 5
    densidade = request.form.get("densidade")

    #abrindo arquivo com o pickle, esse arquivo .pkl foi gerado com o mlflow
    modelar = pickle.load(open("./model.pkl", "rb"))

    #utilizando o modelo para prever e utilizando uma lista com as features
    predicao = modelar.predict([[biraids,age,shapemass,margemass,densidade]])
    
    # O valor irá prever 0 ou 1
    return (f"""Gravidade Câncer de Mama: {str(predicao)}
    | LEITURA:
    0 = Benigno
    1 = Maligno
    """)



#rodando no meu arquivo principal
if __name__=="__main__":
    #port =int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=5000)