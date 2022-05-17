import pandas as pd

import matplotlib.pyplot as plt
#import mlflow

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


datawoman = pd.read_csv("mammographic_masses.data", header=None)
datawoman.head()

datawoman.head()

datawoman.shape

datawoman.isnull().sum()


for coluna in datawoman.columns:
    print(f'{coluna}: {datawoman[coluna].unique()}')

datawoman.info()


datawoman.columns = ["birads", "age", "shapemass", "margemass", "densidade", "gravidade"]
def aplicar_mediana():
    for coluna in datawoman.columns:
        moda = datawoman[coluna].mode()[0]
        datawoman[coluna] = datawoman[coluna].replace("?",moda)
aplicar_mediana()

def verificar_unicos():
    for coluna in datawoman.columns:
        print(f'{coluna}: {datawoman[coluna].unique()}')
verificar_unicos()

def muda_tipo():
    for coluna in datawoman.select_dtypes(include="object"):
        datawoman[coluna] = datawoman[coluna].astype("int32")
muda_tipo()

datawoman.info()

previsores = datawoman.iloc[:,0:5].values
classe =datawoman.iloc[:,5].values

X_train,X_test,y_train,y_test = train_test_split(previsores, classe, test_size=0.3, random_state=42)

c1 = GaussianNB()
c2 = DecisionTreeClassifier(min_samples_leaf=6)
c3 = RandomForestClassifier(n_estimators=5000,random_state=42, min_samples_leaf=6)
c4 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.08, random_state=42, max_depth=4)
clfs =  [c1,c2,c3,c4]

# def criador_modelo():
#     for clf in clfs:
#         modelo = clf.fit(X_train, y_train)
#         previsao = modelo.predict(X_test)
#         acuracia= accuracy_score(y_test, previsao)
#         precisao = precision_score(y_test, previsao)
#         revocacao = recall_score(y_test, previsao)
#         f1score = f1_score(y_test, previsao)

#         print(f"""
#         Modelo: {clf}
#         Acurácia: {acuracia}
#         Precisão: {precisao}
#         Recall: {revocacao}
#         F1-score: {f1score}
#         """)

# criador_modelo()


# mlflow.set_experiment("Mamograph")
# with mlflow.start_run():
#     c1 = GaussianNB()
#     c2 = DecisionTreeClassifier(min_samples_leaf=6)
#     #estiamdor
#     ms = 6
#     c3 = RandomForestClassifier(n_estimators=5000,random_state=42, min_samples_leaf=6)
#     #estimadores
#     ne=1000
#     mrs=6
#     c4 = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.09, random_state=42, max_depth=4)
#     #estimadores
#     nxe=1000
#     lr=0.01
#     md=4

#     clfs =  [c1,c2,c3,c4]
#     contador=0
#     contar=0
#     for clf in clfs:
#         modelo = clf.fit(X_train, y_train)
#         previsao = modelo.predict(X_test)
#         acuracia= accuracy_score(y_test, previsao)
#         precisao = precision_score(y_test, previsao)
#         revocacao = recall_score(y_test, previsao)
#         f1score = f1_score(y_test, previsao)
#         confusao = confusion_matrix(y_test, previsao)

#         print(f"""
#         Modelo: {clf}
#         Acurácia: {acuracia}
#         Precisão: {precisao}
#         Recall: {revocacao}
#         F1-score: {f1score}
#         """)
#         #parametros
        
#         if contador==1:
#             mlflow.log_param("min_samples_leaf", ms)
#         elif contador ==2:
#             mlflow.log_param("n_estimators", ne)
#             mlflow.log_param("min_sampels_leaf", mrs)
            
#         elif contador==3:
#             mlflow.log_param("n_estimators", nxe)
#             mlflow.log_param("learning_rate", lr)
#             mlflow.log_param("max-depth", md)
#         contador+=1
        
#         categorias = ["benign", "malignant"]
#         confusao = ConfusionMatrixDisplay(confusao,display_labels=categorias)
#         confusao.plot()
#         plt.savefig("confusao.png")
#         mlflow.log_artifact("confusao.png")


#         #metricas
#         mlflow.log_metric("acuracia", acuracia)
#         mlflow.log_metric("precisao", precisao)
#         mlflow.log_metric("recall", revocacao)
#         mlflow.log_metric("f1score", f1score)


#         #salvando imagens
       
      
#         #modelo
        
#         cla = ["NaiveBayes", "DecisionTress", "RandomForest", "GradientBoost"]
    
#         mlflow.sklearn.log_model(modelo,cla[contar])
#         print("Modelo: ", mlflow.active_run().info.run_uuid)
#         contar+=1

# mlflow.end_run()




