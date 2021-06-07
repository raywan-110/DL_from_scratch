import numpy as np
from abc import ABCMeta, abstractmethod

import os
import struct
import array

def parse_mnist(fd):
    DATA_TYPES = {
        0x08: 'B',  # unsigned byte
        0x09: 'b',  # signed byte
        0x0b: 'h',  # short (2 bytes)
        0x0c: 'i',  # int (4 bytes)
        0x0d: 'f',  # float (4 bytes)
        0x0e: 'd'  # double (8 bytes)
    }

    # 1.解析文件信息
    header = fd.read(4)  # 读入四个字节(32bit magic number)
    if len(header) != 4:
        raise ValueError(
            'Invalid IDX file, file empty or does not contain a full header.')
    zeros, data_type, num_dimensions = struct.unpack(
        '>HBB', header)  # 解析字符串(分别为16bit 8bit 8bit)
    if zeros != 0:
        raise ValueError(
            'Invalid IDX file, file must start with two zero bytes. '
            'Found 0x%02x' % zeros)
    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise ValueError('Unknown data type 0x%02x in IDX file' % data_type)

    # 2.读取数据集维度信息
    dimension_sizes = struct.unpack(
        '>' + 'I' * num_dimensions,
        fd.read(4 * num_dimensions))  # 以   32bit方式读入描述数据集维度的信息

    # 3.读取并转化数据集
    data = array.array(data_type, fd.read())
    data.byteswap()  # 转化为intel支持的小字节序
    return np.array(data).reshape(dimension_sizes)


def load_mnist(path):
    '''返回训练数据集与测试数据集'''
    # 解析并处理训练集
    fd = open(os.path.join(path, 'train-images-idx3-ubyte'), 'rb')
    X_train = parse_mnist(fd)
    fd.close()
    fd = open(os.path.join(path, 'train-labels-idx1-ubyte'), 'rb')
    Y_train = parse_mnist(fd)
    fd.close()
    # 解析并处理测试集
    fd = open(os.path.join(path, 't10k-images-idx3-ubyte'), 'rb')
    X_test = parse_mnist(fd)
    fd.close()
    fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'), 'rb')
    Y_test = parse_mnist(fd)
    fd.close()
    return X_train, Y_train, X_test, Y_test


class network(object):
    '''定义net网络类,要点：
       1. 采用行模式存储权重和偏差
       2. 权重和偏差利用broadcast机制相加'
       3. 保留激活值与其净输入值
       4. 反向传播输出的是一个batch的平均梯度'''

    def __init__(self, sizes, act):
        self.num_layers = len(sizes)  # 网络的深度
        self.sizes = sizes
        self.bias = [np.zeros((1, y)) for y in sizes[1:]]  # bias默认初始化策略
        self.weights = [
            np.random.randn(x, y) / np.sqrt(y)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]  # weights默认初始化策略
        self.activations = []  # 保存激活值，用于反向传播[第一层(原始特征)到倒数第二层]
        self.pure_output = []  # 保存净输出[第二层到倒数第二层]
        self.train = True  # 表示是否为训练状态
        if act == 'sigmoid':
            self.activation = self.sigmoid
            self.actication_prime = self.sigmoid_prime
        elif act == "relu":
            self.activation = self.relu
            self.actication_prime = self.relu_prime
        elif act == "tanh":
            self.activation = self.tanh
            self.actication_prime = self.tanh_prime

    '''状态切换'''
    def is_eval(self):
        self.train = False

    def is_train(self):
        self.train = True

    '''激活函数与其导数'''
    def sigmoid(self, z):
        # 解决溢出问题
        # 把大于0和小于0的元素分别处理
        # 原来的sigmoid函数是 1/(1+np.exp(-Z))
        # 当Z是比较小的负数时会出现上溢，此时可以通过计算exp(Z) / (1+exp(Z)) 来解决
        mask = (z > 0)
        positive_out = np.zeros_like(z, dtype='float64')
        negative_out = np.zeros_like(z, dtype='float64')
        # 大于0的情况
        positive_out = 1 / (1 + np.exp(-z, positive_out, where=mask))
        # 清除对小于等于0元素的影响
        positive_out[~mask] = 0
        # 小于等于0的情况
        expZ = np.exp(z, negative_out, where=~mask)
        negative_out = expZ / (1 + expZ)
        # 清除对大于0元素的影响
        negative_out[mask] = 0
        return positive_out + negative_out

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def tanh(self, z):
        return np.tanh(z)

    def tanh_prime(self, z):
        return (1 - np.power(z, 2))  # overflow 

    def relu(self, z):
        return np.maximum(z, 0)

    def relu_prime(self, z):
        return np.where(z < 0, 0, 1)  # z小于0则为0，否则为1

    '''前向传播'''
    def forward(self, a):
        # 传播前清空列表
        self.activations.clear()
        self.pure_output.clear()
        if self.train:
            self.activations.append(a)  # 输入作为第一层激活值
        for b, w in zip(self.bias[:-1], self.weights[:-1]):
            z = np.dot(a, w) + b  # 净输入
            a = self.activation(z)  # 激活值
            if self.train:
                self.activations.append(a)  # 保存中间激活值
                self.pure_output.append(z)  # 保存中间净输出
        z_last = np.dot(a,
                        self.weights[-1]) + self.bias[-1]  # 最后的输出层(激活函数与损失有关)
        return z_last

    '''反向传播'''
    def backward(self, z_last, loss, label):
        # 准备好存放梯度的容器
        batch_size = z_last.shape[0]
        grad_b = [np.zeros(b.shape) for b in self.bias]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # 计算最后一层的误差值(损失函数对最后一层净输出的导数)
        delta = loss.func_derivative(z_last, label)
        # 计算最后一层的权重与偏差梯度
        grad_b[-1] = delta.mean(0, keepdims=True)  # 平均梯度
        grad_w[-1] = np.dot(self.activations[-1].transpose(),
                            delta) / np.float(
                                batch_size)  # activations[-1]存放的是倒数第二层的激值
        # 去除输出层激活值，从倒数第二层开始反向传播
        for l in range(2, self.num_layers):
            grad_a2z = self.actication_prime(
                self.pure_output[-l + 1])  # 计算本层的激活函数对净输出的导数
            delta = grad_a2z * np.dot(
                delta,
                self.weights[-l + 1].transpose())  # weights比保存的activation刚好多一层
            grad_b[-l] = delta.mean(0, keepdims=True)
            grad_w[-l] = np.dot(self.activations[-l].transpose(),
                                delta) / np.float(batch_size)
        return grad_b, grad_w


class Loss(object):
    __metaclass__ = ABCMeta

    # 定义损失函数
    @abstractmethod
    def func(self):
        pass

    # 定义损失函数的导数
    @abstractmethod
    def func_derivative(self):
        pass


class squared_loss(Loss):
    def func(self, z_last, y):
        return (z_last - y.reshape(z_last.shape))**2 / 2

    def func_derivative(self, z_last, y):
        return z_last - y.reshape(y.hat.shape)


class cross_entropy_loss(Loss):
    def func(self, z_last, y):
        z_max = np.max(z_last, 1, keepdims=True)
        z_exp = np.exp(z_last - z_max)
        partition = z_exp.sum(1, keepdims=True)
        activations = z_exp / partition
        #         print('activations:\n',activations,'\nsum:\n',activations.sum(1))
        correct_logprobs = -np.log(5e-5 + activations[range(len(z_last)), y])
        return 1. / len(z_last) * np.sum(correct_logprobs)  # 防止下溢出

    def func_derivative(self, z_last, y):
        z_max = np.max(z_last, 1, keepdims=True)
        z_exp = np.exp(z_last - z_max)
        partition = z_exp.sum(1, keepdims=True)
        activations = z_exp / partition
        activations[range(len(z_last)), y] -= 1  # 这里是以one-hot码定义的损失
        return activations


class optim(object):
    __metaclass__ = ABCMeta

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    @abstractmethod
    def step(self):
        pass


class SGD(optim):
    def __init__(self, params, lr, weight_decay=None):
        self.params = params
        self.lr = lr
        self.weight_decay = None
        if weight_decay is not None:
            self.weight_decay = weight_decay

    def step(self, grad):
        # 准备好原始权重
        old_bias = self.params[0]
        old_weights = self.params[1]
        # 准备好本轮梯度
        grad_b = grad[0]
        #         print('grad_b:\n',grad_b)
        grad_w = grad[1]
        #         print('grad_w:\n',grad_w)
        # 更新参数
        if self.weight_decay is not None:
            new_bias = [
                old_b - self.lr * gb for old_b, gb in zip(old_bias, grad_b)
            ]
            # 考虑正则化的梯度
            new_weights = [
                old_w - self.lr * (gw + self.weight_decay * old_w)
                for old_w, gw in zip(old_weights, grad_w)
            ]
        else:
            new_bias = [
                old_b - self.lr * gb for old_b, gb in zip(old_bias, grad_b)
            ]
            new_weights = [
                old_w - self.lr * gw for old_w, gw in zip(old_weights, grad_w)
            ]
        # 保存好新的参数
        self.params = (new_bias, new_weights)
        return new_bias, new_weights


def evaluate(net, test_data):
    # 转变为测试模式
    test_X = test_data[0]
    test_Y = test_data[1]
    net.is_eval()
    z_last = net.forward(test_X)
    z_max = np.max(z_last, 1, keepdims=True)
    exp_scores = np.exp(z_last - z_max)  # 防止上溢出
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    test_results = np.argmax(probs, axis=1)
    net.is_train()
    return int(sum(test_results == test_Y)) / float(len(test_Y))


def BGD_train_test(net, optimizer, loss, train_data, epochs, test_data):
    X_train = train_data[0]
    Y_train = train_data[1]
    for i in range(epochs):
        # 前向传播
        z_last = net.forward(X_train)
        # 反向传播
        grad = net.backward(z_last, loss, Y_train)
        # 更新参数
        new_params = optimizer.step(grad)
        net.bias = new_params[0]
        net.weights = new_params[1]
        # 评估
        if i % 5 == 0:
            print('epoch', i + 1)
            print('loss: {:.4f}'.format(loss.func(z_last, Y_train)))
            res = evaluate(net, (X_train, Y_train))
            print('acc. in train set', res)
            print('acc. in test set', evaluate(net, test_data))


def MBGD_train_test(net, optimizer, loss, train_data, epochs, batch_size,
                    test_data):
    # 获取训练数据
    X_train = train_data[0]
    Y_train = train_data[1]
    # 获取训练集长度
    n_train = len(X_train)  # 获取训练数据长度
    for i in range(epochs):
        # 打乱数据集
        indices = np.random.permutation(n_train)
        X_train[:] = X_train[indices]
        Y_train[:] = Y_train[indices]
        # 准备mini_batch的数据(由于shuffle因素存在，理论上数据都可以被选中)
        X_batches = [
            X_train[k:k + batch_size] for k in range(0, n_train, batch_size)
            if k + batch_size <= n_train
        ]
        Y_batches = [
            Y_train[k:k + batch_size] for k in range(0, n_train, batch_size)
            if k + batch_size <= n_train
        ]
        # 训练一个epoch
        L = []
        for j in range(len(X_batches)):
            data = X_batches[j]
            label = Y_batches[j]
            # forward
            z_last = net.forward(data)
            l_value = loss.func(z_last, label)
            L.append(l_value)
            # backward
            grad = net.backward(z_last, loss, label)
            # update params
            new_params = optimizer.step(grad)
            net.bias = new_params[0]
            net.weights = new_params[1]
        print("Epoch {}: loss: {:.4f}".format(i + 1, np.mean(L)))
        if (i % 5 == 0):
            # 观察测试集上的准确度
            print("Epoch {}: accuracy in test set: {:.4f}".format(
                i, evaluate(net, test_data)))


if __name__ == "__main__":
    # 加载数据集
    path = '../data/MNIST/raw'
    train_images, train_labels, test_images, test_labels = load_mnist(path)

    n_train, w, h = train_images.shape
    X_train = train_images.reshape((n_train, w * h))
    Y_train = train_labels

    n_test, w, h = test_images.shape
    X_test = test_images.reshape((n_test, w * h))
    Y_test = test_labels

    print('X_train shape:',X_train.shape)
    print('Y_train shape',Y_train.shape)
    print('X_test shape',X_test.shape)
    print('Y_test shape',Y_test.shape)

    # 数据归一化
    X_train = (X_train.astype(float) - 128.0) / 128.0
    X_test = (X_test.astype(float) - 128.0) / 128.0
    # X_train = X_train.astype(float) / 255.
    # X_test = X_test.astype(float) / 255.
    train_data = (X_train, Y_train)
    test_data = (X_test, Y_test)

    # 定义超参数
    epochs = 100
    batch_size = 128
    lr = 0.01

    # 定义模型与优化器
    net = network(sizes=[784, 512, 10], act='relu')
    optimizer = SGD((net.bias, net.weights), lr,weight_decay=5e-4)

    # 定义损失函数
    loss = cross_entropy_loss()

    # 批量梯度下降训练/测试模型
    # BGD_train_test(net,optimizer,loss,train_data,epochs,test_data)

    # 小批量梯度下降训练/测试模型
    MBGD_train_test(net, optimizer, loss, train_data, epochs, batch_size,
                    test_data)

    # 最终测试
    print("final test: accuracy in test set: {:.4f}".format(evaluate(net, test_data)))
