from flask import Flask, render_template, request
import pickle
import os


app = Flask(__name__)


@app.route("/")
@app.route("/index")
def pagina():
    return render_template("index.html")


@app.route("/prever", methods=["POST"])
def predicao():
    biraids = request.form.get("biraids")
    age =request.form.get("age")
    shapemass = request.form.get("shapemass")
    margemass = request.form.get("margemass")
    densidade = request.form.get("densidade")
    modelar = pickle.load(open("./model.pkl", "rb"))
    predicao = modelar.predict([[biraids,age,shapemass,margemass,densidade]])
    return (f"""Gravidade Câncer de Mama: {str(predicao)}
    | LEITURA:
    0 = Benigno
    1 = Maligno
    """)




if __name__=="__main__":
    #port =int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=5000)