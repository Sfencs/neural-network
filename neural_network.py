import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork():

    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate

        # 链接权重矩阵
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        # 使用正态分布采样权重
        # self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        # self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        # 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self,inputs_list,targets_list):

        # 初始化为array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)

        # 更新权重
        # 隐藏层与输出层
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        # 输入层与隐藏层
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))

    def query(self,inputs_list):

        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs




if __name__ == '__main__':

    input_nodes = 784 # 28x28
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3


    n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

    with open('mnist_train_100.csv','r') as f:
        train_list = f.readlines()

    for record in train_list:


        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

    with open('mnist_test_10.csv', 'r') as f:
        test_list = f.readlines()

    for record in test_list:

        all_values = record.split(',')
        image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
        matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
        matplotlib.pyplot.show()

        result = n.query((numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01)
        print(str(numpy.argmax(result)),str(all_values[0]))






