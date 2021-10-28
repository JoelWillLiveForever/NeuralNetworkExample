import numpy as np
import scipy.special as ss

class neuralNetwork:

    # инициализация сети
    def __init__(self, input, hidden, output, learningRate) -> None:
        self.input = input
        self.hidden = hidden
        self.output = output
        self.learningRate = learningRate

        # веса
        self.w_ih = np.random.rand(self.hidden, self.input) - 0.5
        self.w_ho = np.random.rand(self.output, self.hidden) - 0.5

        # функция активации
        self.activation_fun = lambda x: ss.expit(x)

    # обучение
    def train(self, input_list, target_list):
        # преобразование списка входящих значений в двумерный массив
        target_data = np.array(target_list, ndmin=2).T
        input_data = np.array(input_list, ndmin=2).T 

        hidden_inputData = np.dot(self.w_ih, input_data)    # входящие сигналы скрытого слоя
        hidden_outputData = self.activation_fun(hidden_inputData)   # исходящие сигналы от скрытого слоя
        
        final_inputData = np.dot(self.w_ho, hidden_outputData)  # входящие сигналы последнего слоя
        final_outputData = self.activation_fun(final_inputData) # исходящие сигналы от последнего слоя

        # определение ошибок на выходе сети
        output_errors = target_data - final_outputData
        hidden_errors = np.dot(self.w_ho.T, output_errors)  # определение ошибок на скрытом слое

        self.w_ho += self.learningRate * np.dot((output_errors * final_outputData * (1.0 - final_outputData)), np.transpose(hidden_outputData))
        self.w_ih += self.learningRate * np.dot((hidden_errors * hidden_outputData * (1.0 - hidden_outputData)), np.transpose(input_data))


    # опрос
    def query(self, input_list):
        input_data = np.array(input_list, ndmin=2).T    # преобразование списка входящих значений в двумерный массив

        hidden_inputData = np.dot(self.w_ih, input_data)    # входящие сигналы скрытого слоя
        hidden_outputData = self.activation_fun(hidden_inputData)   # исходящие сигналы от скрытого слоя
        
        final_inputData = np.dot(self.w_ho, hidden_outputData)  # входящие сигналы последнего слоя
        final_outputData = self.activation_fun(final_inputData) # исходящие сигналы от последнего слоя

        return final_outputData

    # информация о сети
    def __str__(self) -> str:
        print(f'Количество входных нейронов:  {self.input}')
        print(f'Количество скрытых нейронов:  {self.hidden}')
        print(f'Количество выходных нейронов: {self.output}')
        print(f'\nКоэффициент обучения: {self.learningRate}')
        print(f'\nВеса между входным и скрытым слоями: {self.w_ih}')
        print(f'\nВеса между скрытым и выходными слоями: {self.w_ho}')