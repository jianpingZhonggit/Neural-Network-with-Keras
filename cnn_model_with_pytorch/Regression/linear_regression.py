import torch
from torch.autograd import Variable
import numpy as np
import random
import matplotlib.pyplot as plt
from torch import nn


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x)
y = 3 * x + 10 + torch.rand(x.size())
print(y)

# 画图
plt.scatter(x.data.numpy(), y.data.numpy())
plt.show()


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, input_data):
        out_ = self.linear(input_data)
        return out_


if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

num_epochs = 1000
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x).cuda()
        target = Variable(y).cuda()
    else:
        inputs = Variable(x)
        target = Variable(y)

    # 前向传播
    out = model(inputs)
    loss = criterion(out, target)

    # 后向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print('Epoch[{}/{}], loss:{:.6f}'.format(epoch+1, num_epochs, loss.item()))
model.eval()
if torch.cuda.is_available():
    predict = model(Variable(x).cuda())
    predict = predict.data.cpu().numpy()
else:
    predict = model(Variable(x))
    predict = predict.data.numpy()
plt.plot(x.numpy(), y.numpy(), 'ro', label='Original Data')
plt.plot(x.numpy(), predict, label='Fitting Line')
plt.show()
