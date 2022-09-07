#librerias para visualizacion de datos 
import numpy as np #instalacion manual librerias
import matplotlib.pyplot as plt #instalacion manual librerias

sigmoid = lambda x: 1/(1 + np.e**-x) #Funciones sigmoide
tanh = 	lambda x: np.tanh(x) #Funcion tangente  con hiperbolica
relu = 	lambda x: np.maximum(0, x)

def random_points(n = 100): #Conjunto de datos y predicciones del perceptron
	x = np.random.uniform(0.0, 1.0, n) #Datos aleatorios
	y = np.random.uniform(0.0, 1.0, n)

	return np.array([x, y]).T #retorna vector columna

class Perceptron:

	def __init__(self, n_inputs, act_f):
		# bias es un valor de direccion que permite cambiar o disparar la función de activación
		'''
		Inicializamos pesos, el bias y la funcion de activacion,
		'''
		self.weights = np.random.rand(n_inputs,1)
		self.bias = np.random.rand()
		self.act_f = act_f
		self.n_inputs = n_inputs

	def predict(self, x):
		'''
		Metodo predict, realiza el producto punto entre
		las entradas y los pesos, suma el bias y evalua en
		la funcion de activacion.
		'''
		return self.act_f(x @ self.weights + self.bias)

	def fit(self, x, y, epochs = 100):#numero de epocas a utilizar
		'''
		Metodo fit, se encarga de entrenar al perceptron,
		calculando el error en cada iteracion y ajustando
		los pesos y el bias.

		Podemos entrenar hasta que el error sea 0,
		pero no es recomendable por que tenemos mucho
		riesgo de sobreajuste de direccion.
		'''
		for i in range(epochs):
			for j in range(len(x)):
				output = self.predict(x[j])
				error = y[j] - output
				self.weights = self.weights + (error * x[j][1]) #cuanto se equivoca respecto a la prediccion del atgoritmo
				self.bias = self.bias + error

def main():
	points = random_points(10000) #numero de puntos para testear
	plt.scatter(points[:,0], points[:,1], s = 10) 
	plt.show()#FIGURA 1 

	'''
	COMPUERTA AND
	'''

	x = np.array([
				[0,0],
				[0,1],
				[1,0],
				[1,1]
	])
	
	y = np.array([
		[0],
		[0],
		[0],
		[1]
	])

	p_and = Perceptron(2, sigmoid) # 2 entradas funcion de activacion y numero de inputs

	yp = p_and.predict(points)
	plt.scatter(points[:,0], points[:,1], s = 10, c=yp, cmap='GnBu') #mapa de colores de green a blue
	plt.show() #FIGURA 2 SIN ENTRENAR 
	plt.savefig('Perceptron sin entrenar')

	p_and.fit(x = x, y = y, epochs=1000) # numero de iteraciones 

	yp = p_and.predict(points)
	plt.scatter(points[:,0], points[:,1], s = 10, c=yp, cmap='GnBu')#mapa de colores de green a blue entrenado
	plt.show() #FIGURA 3 ENTRENADO
	plt.savefig('Perceptron entrenado')

if __name__ == '__main__':
	main()
	
