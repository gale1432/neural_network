import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class WeightInferenceSystem:
    def __init__(self):
        ## FUZZY SETS PARA INPUT DE PESOS
        self.current_weight = ctrl.Antecedent(np.arange(-0.8, 0.8, 0.01), 'current_weight')
        self.weight_change = ctrl.Antecedent(np.arange(-0.08, 0.08, 0.001), 'weight_change')
        ## SET FUZZY PARA OUTPUT DE PESOS
        self.next_weight = ctrl.Consequent(np.arange(-0.8, 0.8, 0.01), 'next_weight')
        ## FUNCIONES DE MEMBRESIA PARA PESOS
        self.current_weight['low'] = fuzz.trimf(self.current_weight.universe, [-0.8, -0.2, 0.4])
        self.current_weight['high'] = fuzz.trimf(self.current_weight.universe, [-0.4, 0.2, 0.8])
        self.weight_change['low'] = fuzz.trimf(self.weight_change.universe, [-0.008, -0.002, 0.004])
        self.weight_change['high'] = fuzz.trimf(self.weight_change.universe, [-0.004, 0.002, 0.008])
        self.next_weight['low'] = fuzz.trimf(self.next_weight.universe, [-0.8, -0.2, 0.4])
        self.next_weight['high'] = fuzz.trimf(self.next_weight.universe, [-0.4, 0.2, 0.8])
        ## REGLAS DE INFERENCIA DE PESOS
        self.regla1 = ctrl.Rule(self.current_weight['low'] & self.weight_change['low'], self.next_weight['low'])
        self.regla2 = ctrl.Rule(self.current_weight['low'] & self.weight_change['high'], self.next_weight['low'])
        self.regla3 = ctrl.Rule(self.current_weight['high'] & self.weight_change['low'], self.next_weight['high'])
        self.regla4 = ctrl.Rule(self.current_weight['high'] & self.weight_change['high'], self.next_weight['high'])
        self.regla5 = ctrl.Rule(self.current_weight['low'], self.next_weight['low'])
        self.regla6 = ctrl.Rule(self.current_weight['high'], self.next_weight['high'])
        ## IMPLEMENTACION DEL SISTEMA DE INFERENCIA DIFUSO
        self.weight_ctrl = ctrl.ControlSystem([self.regla1, self.regla2, self.regla3, self.regla4, self.regla5, self.regla6])
        self.weight_sim = ctrl.ControlSystemSimulation(self.weight_ctrl)

    def weight_inference_system(self, current_weight_network, weight_change_network):
        ## OBTENCION DE NUEVOS PESOS
        self.weight_sim.input['current_weight'] = current_weight_network
        self.weight_sim.input['weight_change'] = weight_change_network
        self.weight_sim.compute()
        return self.weight_sim.output['next_weight']

    def weight_change_all(self, current_weight_matrix, change_matrix):
        next_weight = []
        width, length = current_weight_matrix.shape
        for i in range(width):
            row_weight = []
            for j in range(length):
                row_weight.append(self.weight_inference_system(current_weight_matrix[i, j], change_matrix[i, j]))
            next_weight.append(np.array(row_weight).astype(np.float64))
        return np.array(next_weight)

class BiasInferenceSystem:
    def __init__(self):
        current_bias = ctrl.Antecedent(np.arange(-0.8, 0.8, 0.01), 'current_bias')
        bias_change = ctrl.Antecedent(np.arange(-0.5, 1.7, 0.001), 'bias_change')
        ## FUZZY SET PARA OUTPUT DE BIAS
        next_bias = ctrl.Consequent(np.arange(-0.8, 0.8, 0.01), 'next_bias')
        ## FUNCIONES DE MEMBRESIA PARA BIAS
        current_bias['low'] = fuzz.gaussmf(current_bias.universe, -0.2, 0.6)
        current_bias['high'] = fuzz.gaussmf(current_bias.universe, 0.2, 0.6)
        bias_change['low'] = fuzz.gaussmf(bias_change.universe, 0.3, 0.5)
        bias_change['high'] = fuzz.gaussmf(bias_change.universe, 0.9, 0.5)
        next_bias['low'] = fuzz.gaussmf(next_bias.universe, -0.2, 0.6)
        next_bias['high'] = fuzz.gaussmf(next_bias.universe, 0.2, 0.6)
        ## REGLAS DE INFERENCIA DE BIAS
        regla1 = ctrl.Rule(current_bias['low'] & bias_change['low'], next_bias['low'])
        regla2 = ctrl.Rule(current_bias['low'] & bias_change['high'], next_bias['low'])
        regla3 = ctrl.Rule(current_bias['high'] & bias_change['low'], next_bias['high'])
        regla4 = ctrl.Rule(current_bias['high'] & bias_change['high'], next_bias['high'])
        regla5 = ctrl.Rule(current_bias['low'], next_bias['low'])
        regla6 = ctrl.Rule(current_bias['high'], next_bias['high'])
        ## IMPLEMENTACION DEL SISTEMA DE INFERENCIA DIFUSO
        bias_ctrl = ctrl.ControlSystem([regla1, regla2, regla3, regla4, regla5, regla6])
        self.bias_sim = ctrl.ControlSystemSimulation(bias_ctrl)

    def bias_inference_system(self, current_bias_network, bias_change_network, last_layer_bias):
        ## FUZZY SETS PARA INPUT DE BIAS
        if last_layer_bias:
            return 0.0
        ## OBTENCION DE NUEVO BIAS
        self.bias_sim.input['bias_change'] = bias_change_network
        self.bias_sim.input['current_bias'] = current_bias_network
        self.bias_sim.compute()
        return self.bias_sim.output['next_bias']

    def bias_change_all(self, current_bias_matrix, bias_change_float, last_layer_bias=False):
        next_bias = []
        width, length = current_bias_matrix.shape
        for i in range(width):
            row_weight = []
            for j in range(length):
                row_weight.append(self.bias_inference_system(current_bias_matrix[i, j], bias_change_float, last_layer_bias))
            next_bias.append(np.array(row_weight).astype(np.float64))
        return np.array(next_bias)

"""W1 = np.random.rand(20, 550) - 0.5
dW1 = (np.random.rand(20, 550) - 0.5)*0.1
print(W1.shape)
wis = WeightInferenceSystem()
print(wis.weight_inference_system(W1, dW1))
print(wis.weight_inference_system(0.5, -0.05))"""