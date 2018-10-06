"""
神经网络步骤
1.初始化函数,输入层,隐藏层,输出层初始化
2.神经网络训练
3.给定输入,输出答案
"""
import numpy
import scipy.special
class nnw():
    def __init__(self, inodes, hnodes, onodes, learningrate):
        # 输入层节点,隐藏层节点,输出层节点,学习率
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        self.learningrate = learningrate
        # 初始化隐藏层和输出层的权重,定义激活函数
        self.wh = numpy.random.rand(inodes, inodes) - 0.5
        self.wo = numpy.random.rand(hnodes, hnodes) - 0.5
        self.act_fun=lambda x: scipy.special.expit(x)
        pass

    def train(self, input_list, target_list):
        train_inputs = numpy.array(input_list, ndmin=2).T
        train_target = numpy.array(target_list, ndmin=2).T
        hidden_input = numpy.dot(self.wh, train_inputs)
        hidden_outputs = self.act_fun(hidden_input)
        final_input = numpy.dot(self.wo, hidden_outputs)
        final_output = self.act_fun(final_input)
        #输出层误差和隐藏层误差
        output_error = train_target - final_output
        hidden_error = numpy.dot(self.wo.T, output_error)
        #权重的调整
        self.wo += self.learningrate * numpy.dot((output_error * final_output * (1.0-final_output)),
                                            numpy.transpose(hidden_outputs))
        self.wh += self.learningrate * numpy.dot((hidden_error * hidden_outputs * (1.0 - hidden_outputs)),
                                                 numpy.transpose(train_inputs))
        pass
    #结果函数
    def query(self, input_arr):
        query_inputs = numpy.array(input_arr, ndmin=2).T
        hidden_input = numpy.dot(self.wh, query_inputs)
        hidden_outputs = self.act_fun(hidden_input)
        final_input = numpy.dot(self.wo, hidden_outputs)
        final_output = self.act_fun(final_input)
        return final_output

nnw_ins = nnw(3, 3, 3, 0.3)
#训练1000次
for i in range(10000):
    nnw_ins.train([1.0, 0.5, 1.5], [0.8, 0.4, 0.6])

#使用训练后的权重值,再次输出
result = nnw_ins.query([1.0, 0.5, 1.5])

print(result)

