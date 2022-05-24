## Detector de Gravidade Câncer de mama com flask e docker:  Instruções

### 1 - Uso abrir sua IDE
### 2 - Construa a imagem: 
 - docker build -t nomedaimagem .
### 3 - Rode a imagem: 
 - Opção 1: docker run -d -p 5000:5000 nomedaimagem
 - Opção 2: docker run -p 127.0.0.1:5000:5000/tcp nomedaimagem

### 4 - Informações dos atributos:

- 6 atributos no total (1 classe, 1 não preditivo, 4 atributos preditivos)

- 1- BI-RADS assessment: 1 a 5 (ordinal, não preditivo)
- 2- Age:idade em anos (integer)
- 3- mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
- 4- mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
- 5- mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
- 6- GFravidade: benigno=0 or maligno=1 (binominal,meta)

### 5 - Arquivos importantes
- Arquivo principal: main.py
- Arquivos necessários para rodar: main.py, templates/, static/, requirements.py e model.pkl
- Arquivos secundários: requirements.txt, Dockerfile, model.pkl, .flaskenv
- Conjuntos de dados: mammographic_masses.data
- Dicionário dos dados: mammographic_masses.names
- Arquivos de manipulção de dados: classifier.ipynb e classifier.py



##### Desenvolvido por Adolfo Matias
