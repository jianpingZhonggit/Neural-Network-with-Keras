import torch


class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.convolution1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.convolution2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.convolution3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.convolution4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.convolution5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256*4*4, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 1000)
        )

    def forward(self, x):
        convolution1_out = self.convolution1(x)
        convolution2_out = self.convolution2(convolution1_out)
        convolution3_out = self.convolution3(convolution2_out)
        convolution4_out = self.convolution4(convolution3_out)
        convolution5_out = self.convolution5(convolution4_out)
        res = convolution5_out.view(convolution5_out.size(0), -1)
        out = self.dense(res)
        return out


# input_data = torch.randn((8, 3, 224, 224))
# model = AlexNet()
# out = model(input_data)
# print(model)



