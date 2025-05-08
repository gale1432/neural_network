import numpy as np
import joblib
import fuzzy_system

class FinalNetwork:
    def __init__(self):
        self.W1 = np.random.rand(20, 550) - 0.5
        self.b1 = np.random.rand(20, 1) - 0.5
        self.W2 = np.random.rand(30, 20) - 0.5
        self.b2 = np.random.rand(30, 1) - 0.5
        self.W3 = np.random.rand(11, 30) -0.5 #11
        self.b3 = np.random.rand(11, 1) - 0.5 #11
        self.weight_system = fuzzy_system.WeightInferenceSystem()
        self.bias_system = fuzzy_system.BiasInferenceSystem()
        self.weight_dict = dict()

    def relu(self, Z):
        return np.maximum(0, Z)

    def deriv_relu(self, Z):
        return Z > 0

    def softmax(self, Z):
        return np.exp(Z)/sum(np.exp(Z))

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.relu(Z2)
        Z3 = self.W3.dot(A2) + self.b3
        #######np.save('examination.npy', Z3)
        A3 = self.softmax(Z3)

        return Z1, A1, Z2, A2, Z3, A3

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def back_prop(self, Z1, A1, Z2, A2, Z3, A3, Y, X):
        m = Y.size
        new_Y = self.one_hot(Y)
        dZ3 = A3 - new_Y
        dW3 = (1/m)*dZ3.dot(A2.T)
        db3 = (1/m)*np.sum(dZ3)
        dZ2 = self.W3.T.dot(dZ3) * self.deriv_relu(Z2)
        dW2 = (1/m)*dZ2.dot(A1.T)
        db2 = (1/m)*np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * self.deriv_relu(Z1)
        dW1 = (1/m)*dZ1.dot(X.T)
        db1 = (1/m)*np.sum(dZ1)
        return dW1, db1, dW2, db2, dW3, db3
    ## METHOD OF UPDATE THAT DOESNT USE FUZZY LOGIC
    """def update_params(self, dW1, db1, dW2, db2, dW3, db3, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2
        self.W3 = self.W3 - alpha * dW3
        self.b3 = self.b3 - alpha * db3"""

    ## METHOD OF UPDATE THAT USES FUZZY INFERENCE SYSTEMS
    def update_params(self, dW1, db1, dW2, db2, dW3, db3, alpha):
        self.W1 = self.weight_system.weight_change_all(self.W1, alpha*dW1)
        self.b1 = self.bias_system.bias_change_all(self.b1, alpha*db1)
        self.W2 = self.weight_system.weight_change_all(self.W2, alpha*dW2)
        self.b2 = self.bias_system.bias_change_all(self.b2, alpha*db2)
        self.W3 = self.weight_system.weight_change_all(self.W3, alpha*dW3)
        self.b3 = self.bias_system.bias_change_all(self.b3, alpha*db3, last_layer_bias=True)

    def save_weights(self, dW1, db1, dW2, db2, dW3, db3, iter_number):
        self.weight_dict[str(iter_number)] = dict()
        self.weight_dict[str(iter_number)]['W1'] = self.W1.tolist()
        self.weight_dict[str(iter_number)]["dW1"] = dW1.tolist()
        self.weight_dict[str(iter_number)]['b1'] = self.b1.tolist()
        self.weight_dict[str(iter_number)]['db1'] = db1.tolist()
        self.weight_dict[str(iter_number)]['W2'] = self.W2.tolist()
        self.weight_dict[str(iter_number)]['dW2'] = dW2.tolist()
        self.weight_dict[str(iter_number)]['b2'] = self.b2.tolist()
        self.weight_dict[str(iter_number)]['db2'] = db2.tolist()
        self.weight_dict[str(iter_number)]['W3'] = self.W3.tolist()
        self.weight_dict[str(iter_number)]['dW3'] = dW3.tolist()
        self.weight_dict[str(iter_number)]['b3'] = self.b3.tolist()
        self.weight_dict[str(iter_number)]['db3'] = db3.tolist()

    def get_predictions(self, A):
        return np.argmax(A, 0)

    def get_accuracy(self, pred, Y):
        #print(pred, Y)
        return np.sum(pred == Y) / Y.size

    def gradient_descent(self, X, Y, alpha, num_iters):
        for i in range(1, num_iters+1):
            Z1, A1, Z2, A2, Z3, A3 = self.forward_prop(X)
            dW1, db1, dW2, db2, dW3, db3 = self.back_prop(Z1, A1, Z2, A2, Z3, A3, Y, X)
            self.save_weights(dW1, db1, dW2, db2, dW3, db3, i)
            self.update_params(dW1, db1, dW2, db2, dW3, db3, alpha)
            if i % 10 == 0:
                print(f"iteration {i}")
                print(f"Accuracy:{self.get_accuracy(self.get_predictions(A3), Y)}")
        #return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        return self.weight_dict