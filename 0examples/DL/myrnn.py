import numpy as np


# 输出单元激活函数
def softmax(x):
    x = np.array(x)
    max_x = np.max(x)
    return np.exp(x-max_x) / np.sum(np.exp(x-max_x))


class myRNN:
    def __init__(self, data_dim, hidden_dim=100, bptt_back=4):
        '''
        data_dim: 词向量维度，即词典长度; 
        hidden_dim: 隐单元维度; 
        bptt_back: 反向传播回传时间长度
        '''
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.bptt_back = bptt_back

        # 初始化权重向量 U， W， V; U为输入权重; W为递归权重; V为输出权重
        self.U = np.random.uniform(-np.sqrt(1.0/self.data_dim), np.sqrt(1.0/self.data_dim), 
                                   (self.hidden_dim, self.data_dim))
        self.W = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), 
                                   (self.hidden_dim, self.hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1.0/self.hidden_dim), np.sqrt(1.0/self.hidden_dim), 
                                   (self.data_dim, self.hidden_dim))
    # 前向传播
    def forward(self, x):
        # 向量时间长度
        T = len(x)

        # 初始化状态向量, s包含额外的初始状态 s[-1]
        s = np.zeros((T+1, self.hidden_dim))
        o = np.zeros((T, self.data_dim))

        for t in range(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))#???
            o[t] = softmax(self.V.dot(s[t]))

        return [o, s]
    
    # 预测输出        
    def predict(self, x):
        o, s = self.forward(x)
        pre_y = np.argmax(o, axis=1)
        return pre_y
    
    # 计算损失， softmax损失函数， (x,y)为多个样本
    def loss(self, x, y):
        cost = 0        
        for i in range(len(y)):
            o, s = self.forward(x[i])
            # 取出 y[i] 中每一时刻对应的预测值
            pre_yi = o[range(len(y[i])), y[i]]
            cost -= np.sum(np.log(pre_yi))

        # 统计所有y中词的个数, 计算平均损失
        N = np.sum([len(yi) for yi in y])
        ave_loss = cost / N

        return ave_loss

    # 求梯度, (x,y)为一个样本
    def bptt(self, x, y):
        dU = np.zeros(self.U.shape)
        dW = np.zeros(self.W.shape)
        dV = np.zeros(self.V.shape)

        o, s = self.forward(x)
        delta_o = o
        delta_o[xrange(len(y)), y] -= 1

        for t in np.arange(len(y))[::-1]:
            # 梯度沿输出层向输入层的传播
            dV += delta_o[t].reshape(-1, 1) * s[t].reshape(1, -1)  # self.data_dim * self.hidden_dim
            delta_t = delta_o[t].reshape(1, -1).dot(self.V) * ((1 - s[t-1]**2).reshape(1, -1)) # 1 * self.hidden_dim

            # 梯度沿时间t的传播
            for bpt_t in np.arange(np.max([0, t-self.bptt_back]), t+1)[::-1]:
                dW += delta_t.T.dot(s[bpt_t-1].reshape(1, -1))
                dU[:, x[bpt_t]] = dU[:, x[bpt_t]] + delta_t

                delta_t = delta_t.dot(self.W.T) * (1 - s[bpt_t-1]**2)

        return [dU, dW, dV]

    # 计算梯度   
    def sgd_step(self, x, y, learning_rate):
        dU, dW, dV = self.bptt(x, y)

        self.U -= learning_rate * dU
        self.W -= learning_rate * dW
        self.V -= learning_rate * dV

    # 训练RNN  
    def train(self, X_train, y_train, learning_rate=0.005, n_epoch=5):
        losses = []
        num_examples = 0

        for epoch in xrange(n_epoch):   
            for i in xrange(len(y_train)):
                self.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples += 1

            loss = self.loss(X_train, y_train)
            losses.append(loss)
            print 'epoch {0}: loss = {1}'.format(epoch+1, loss)
            # 若损失增加，降低学习率
            if len(losses) > 1 and losses[-1] > losses[-2]:
                learning_rate *= 0.5
                print 'decrease learning_rate to', learning_rate
