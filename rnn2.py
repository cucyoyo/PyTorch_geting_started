import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step / image height
INPUT_SIZE = 1      # rnn input size / image width
LR = 0.02           # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data

# ---------------------数据-------------------------------------------
# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()

#---------------------搭建RNN----------------------------------------------
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(  # 这回一个普通的 RNN 就能胜任
            input_size=1,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # 有几层 RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):  # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)
       
        r_out, h_state = self.rnn(x, h_state)   # h_state 也要作为 RNN 的一个输入
        # 每一层的输出放到outs[]中
        outs = []    # 保存所有时间点的预测值
        for time_step in range(r_out.size(1)):    # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :]))
        # .stack将list变成tersor
        return torch.stack(outs, dim=1), h_state
    # ---------第二种forward方法：方法二-----------------------------------------
# 其实熟悉 RNN 的朋友应该知道, forward 过程中的对每个时间点求输出还有一招使得计算量比较小的.
# 不过上面的内容主要是为了呈现 PyTorch 在动态构图上的优势, 所以我用了一个 for loop 来搭建那套输出系统. 
# 下面介绍一个替换方式. 使用 reshape 的方式整批计算.
#     def forward(self, x, h_state):
#         r_out, h_state = self.rnn(x, h_state)
#         r_out_reshaped = r_out.view(-1, HIDDEN_SIZE) # to 2D data
#         outs = self.linear_layer(r_out_reshaped)
#         outs = outs.view(-1, TIME_STEP, INPUT_SIZE)  # to 3D data


rnn = RNN()
print(rnn)

#-------------------------训练--------------------------------------
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(60):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    # 取一小段距离，在这距离中取time_step这么多数据点，放到sin和cos中去
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)*2    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)*3
    # 将一维的x、y变成三维并包裹在variable中
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    # h_state初始值在for循环前面赋值为None，之后每一步的h_state都是上一步产生的h_state,用下一步将包裹到variable中才能在此处理
    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(); plt.pause(0.05)
    
#     test_output = rnn(test_x)                   # (samples, time_step, input_size)
    pred_y = prediction.data.numpy().flatten()
    accuracy = sum(pred_y == y_np) / float(y_np.size)
    print('step: ', step, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
           

plt.ioff()
plt.show()

