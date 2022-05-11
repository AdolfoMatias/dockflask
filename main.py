from flask import Flask, render_template, request
import os

app = Flask(__name__)
@app.route("/")
@app.route("/index")

def pagina():
    return render_template("index.html")

if __name__=="__main__":
    port =int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)