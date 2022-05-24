## Detector de Gravidade Câncer de mama com flask e docker:  Instruções

### 1 - Uso abrir sua IDE
### 2 - Construa a imagem: 
 - docker build -t nomedaimagem .
### 3 - Rode a imagem: 
 - Opção 1: docker run -d -p 5000:5000 nomedaimagem
 - Opção 2: docker run -p 127.0.0.1:5000:5000/tcp nomedaimagem

##### Desenvolvido por Adolfo Matias
