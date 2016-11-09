# -*-coding:utf-8-*-
import numpy as np
import operator

def loadData(filename):
    fr = open(filename)
    arrayLine = fr.readlines()
    numberLines = len(arrayLine)
    feature = np.zeros((numberLines, 4))
    labels = np.zeros((numberLines, 1))
    index = 0
    for line in arrayLine:
        curLine = line.strip().split()
        floatLine = [float(i) for i in curLine]
        feature[index, :] = floatLine[0:4]
        labels[index, :] = floatLine[-1]
        index += 1
    return feature, labels

def sigmoid(x):
    #构造S函数
    return 1/(1+np.exp(-x))

def sigmoid_deri(x):
    #构造S函数的导数
    return x * (1 - x)

def test():
    nn = BPNeuralNetwork()
    train_feature, train_labels = loadData("trainData.txt")
    nn.setup(4, 10, 1, 50)
    nn.train(train_feature, train_labels)
    test_feature, test_labels = loadData("testData.txt")
    exp_labels = nn.predict(test_feature)
    res = [exp_labels,test_labels]
    print(res)

class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.hidden_weights = []
        self.input_correction = []
        self.hidden_correction = []

    def setup(self, ni, nh, no, nums):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no

        #初始化各层输入
        self.input_cells = np.ones((nums, self.input_n))
        self.hidden_cells = np.ones((nums, self.hidden_n))
        self.output_cells = np.ones((nums, self.output_n))

        #初始化权重矩阵
        self.input_weights = np.random.uniform(-0.2, 0.2, size=(self.input_n, self.hidden_n))
        self.hidden_weights = np.random.uniform(-2.0, 2.0, size=(self.hidden_n, self.output_n))

        #初始化权重修改矩阵
        self.input_correction = np.zeros((self.input_n, self.hidden_n))
        self.hidden_correction = np.zeros((self.hidden_n, self.output_n))

    def predict(self, inputs):
        self.input_cells[:, :self.input_n-1] = np.array(inputs)

        #计算隐藏层输出
        self.hidden_cells = sigmoid(np.dot(self.input_cells, self.input_weights))

        #计算输出层输出
        self.output_cells = sigmoid(np.dot(self.hidden_cells, self.hidden_weights))

        return self.output_cells

    def back_propagate(self, cases, labels, learn, correct):
        # 前向反馈
        self.predict(cases)

        #获取输出层误差
        output_deltas = (labels - self.output_cells) * sigmoid_deri(self.output_cells)
        #获取隐藏层误差
        back_error = np.dot(output_deltas, self.hidden_weights.T)
        hidden_deltas = back_error * sigmoid_deri(self.hidden_cells)

        #更新输出权重
        change_hidden = np.dot(self.hidden_cells.T, output_deltas)
        self.hidden_weights += learn * change_hidden + correct * self.hidden_correction
        self.hidden_correction = change_hidden

        #更新隐藏层权值
        change_input = np.dot(self.input_cells.T, hidden_deltas)
        self.input_weights += learn * change_input + correct * self.input_correction
        self.input_correction = change_input

        #计算全局误差
        error = sum(0.5 * (labels - self.output_cells)**2)
        return error

    def train(self, case, labels, limit=10000, learn=0.05, correct=0.1, error = 0.001):
        for i in range(limit):
            error_cur = self.back_propagate(case, labels, learn, correct)
            if(error_cur < error):
                break


if __name__ == '__main__':
    test()
















