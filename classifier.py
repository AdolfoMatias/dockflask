#importando as bibliotecas necessárias
import pandas as pd

import matplotlib.pyplot as plt
import mlflow

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



#carregando a base de dados de mamografia
datawoman = pd.read_csv("mammographic_masses.data", header=None)
datawoman.head()

# ## 1  Análise Exploratória de Dados

#vendo o cabeçalho do conjunto de dados
datawoman.head()

#formato do dado
datawoman.shape

#valores NA
datawoman.isnull().sum()

#Veja que não é verdade alguns dados apresentam ?  irei substituir pela mediana logo
for coluna in datawoman.columns:
    print(f'{coluna}: {datawoman[coluna].unique()}')

#informação da nossa tabela
datawoman.info()

# ## 2 Pré-Processamento

#renomenado colunas sem nome
datawoman.columns = ["birads", "age", "shapemass", "margemass", "densidade", "gravidade"]

#moda em dados que possuem valores faltantes ou ?
def aplicar_moda():
    for coluna in datawoman.columns:
        moda = datawoman[coluna].mode()[0]
        datawoman[coluna] = datawoman[coluna].replace("?",moda)
aplicar_moda()

#verificar valores unicos para ver se a mudança ocorreu
def verificar_unicos():
    for coluna in datawoman.columns:
        print(f'{coluna}: {datawoman[coluna].unique()}')
verificar_unicos()

#codificando os dados em dummies
def muda_tipo():
    for coluna in datawoman.select_dtypes(include="object"):
        datawoman[coluna] = datawoman[coluna].astype("int32")
muda_tipo()

#verificando as informações do conjunto de dados
datawoman.info()

# ## 3 Separando previsores e classe e treino/teste

#previsores são as features que usaremos para prever a classe
previsores = datawoman.iloc[:,0:5].values
classe =datawoman.iloc[:,5].values

#Separação de dados em treino e teste  com random_STATE=42
X_train,X_test,y_train,y_test = train_test_split(previsores, classe, test_size=0.3, 
random_state=42)

# ## 4 Criação de Modelo e Predição

#Criando Vários Classificadores: Naive Bayes, Arvores de Decisão, Floresta de Decisão e Gradiente Boosting
c1 = GaussianNB()
c2 = DecisionTreeClassifier(min_samples_leaf=6)
c3 = RandomForestClassifier(n_estimators=5000,random_state=42, min_samples_leaf=6)
c4 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.08, random_state=42, max_depth=4)
#passando uma lista dos classificadores
clfs =  [c1,c2,c3,c4]

def criador_modelo():
    #criando um laço que roda sobre os classificadores
    for clf in clfs:
        modelo = clf.fit(X_train, y_train)
        previsao = modelo.predict(X_test)
        acuracia= accuracy_score(y_test, previsao)
        precisao = precision_score(y_test, previsao)
        revocacao = recall_score(y_test, previsao)
        f1score = f1_score(y_test, previsao)

        print(f"""
        Modelo: {clf}
        Acurácia: {acuracia}
        Precisão: {precisao}
        Recall: {revocacao}
        F1-score: {f1score}
        """)

criador_modelo()

#utilizando o mlflow para colocar todos parametros, imagens e modelos no mlops
mlflow.set_experiment("Mamograph")
with mlflow.start_run():
    c1 = GaussianNB()
    c2 = DecisionTreeClassifier(min_samples_leaf=6)
    #estimador
    ms = 6
    c3 = RandomForestClassifier(n_estimators=5000,random_state=42, min_samples_leaf=6)
    #estimadores
    ne=1000
    mrs=6
    c4 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.09, random_state=42, max_depth=4)
    #estimadores
    nxe=1000
    lr=0.01
    md=4


    #cirnaod lista de classificadores
    clfs =  [c1,c2,c3,c4]

    #valores irão ser usados como contadores no laço
    contador=0
    contar=0

    #iterando sobre o laço de classificadores
    for clf in clfs:
        modelo = clf.fit(X_train, y_train)
        previsao = modelo.predict(X_test)
        acuracia= accuracy_score(y_test, previsao)
        precisao = precision_score(y_test, previsao)
        revocacao = recall_score(y_test, previsao)
        f1score = f1_score(y_test, previsao)
        confusao = confusion_matrix(y_test, previsao)

        print(f"""
        Modelo: {clf}
        Acurácia: {acuracia}
        Precisão: {precisao}
        Recall: {revocacao}
        F1-score: {f1score}
        """)

        #parametros, o contador signifca o numero do laço pois o classificador tem parametros que diferem então personaliza-se o que o modelo tem de acordo com o laço
        
        #Parametros salvos da Arvore de decisão
        if contador==1:
            mlflow.log_param("min_samples_leaf", ms)
        
        #Floresta de decisão parametros
        elif contador ==2:
            mlflow.log_param("n_estimators", ne)
            mlflow.log_param("min_sampels_leaf", mrs)
        
        #Gradiente boosting parametros
        elif contador==3:
            mlflow.log_param("n_estimators", nxe)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("max-depth", md)
        contador+=1

        #gerando matriz de confusão e imagems
        categorias = ["benign", "malignant"]
        confusao = ConfusionMatrixDisplay(confusao,display_labels=categorias)
        confusao.plot()

        plt.savefig("confusao.png")
        #salvando a imagem
        mlflow.log_artifact("confusao.png")


        #salvando as metricas
        mlflow.log_metric("acuracia", acuracia)
        mlflow.log_metric("precisao", precisao)
        mlflow.log_metric("recall", revocacao)
        mlflow.log_metric("f1score", f1score)


        
       
      
        #modelo criando  o cla[contar] é o valor do laço iterado dentro dos classificadores para normear o modelo
        
        cla = ["NaiveBayes", "DecisionTress", "RandomForest", "GradientBoost"]
    
        mlflow.sklearn.log_model(modelo,cla[contar])
        print("Modelo: ", mlflow.active_run().info.run_uuid)
        contar+=1

mlflow.end_run()

# ## Referências

# https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass


