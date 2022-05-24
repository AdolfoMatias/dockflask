from abc import ABC, abstractclassmethod
class NovoModelo(ABC):
	def __init__(self, nome, dados):
		self.__nome = nome
		self.__dados = dados
	@property
	def nome(self):
		return self.__nome
	@nome.setter
	def nome(self,valor):	
		self.__nome = valor
		self.__nome

	@abstractclassmethod
	def falar(self):
		pass
    
class FilhaNovo(NovoModelo):

    def __init__(self,nome, dados,novo):
        super().__init__(nome,dados)
        self.novo=novo

    def falar(self):
        print(f"{self.nome} falou oi")

    @staticmethod
    def triste():
        print("eu nem deveria estar aqui")

if __name__ == "__main__":
    classe = FilhaNovo("Adolfo", "30", 14)
    classe.falar()
    classe.triste()
		