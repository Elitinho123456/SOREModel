import numpy as np

class Perceptron:                           # eta = taxa de aprendizado n_inter = numero de interacoes
    def __init__(self, eta, n_inter):
        self.eta = eta
        self.n_inter = n_inter
        self.pesos = None
        self.bias = None
        self.errors = []

        self.predict = lambda x: np.heaviside(x, 0)

        pass

    def fit(self, X, y):
        self.pesos = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_inter):

            error = 0

            self.errors.append(np.sum(error))

            for X_i, alvo in zip(X, y):
                net_input = np.dot(X_i, self.pesos) + self.bias
                previsao = self.predict(net_input)
                erro = alvo - previsao

                self.pesos += self.eta * erro * X_i
                self.bias += self.eta * erro

                self.error.append(int(erro != 0))
