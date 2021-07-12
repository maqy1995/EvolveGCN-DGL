import torch

torch.manual_seed(123)


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.toy_model = ToyModel(weight=True)

    def forward(self, x):
        W = self.toy_model.weight
        # W = torch.exp(W)
        self.toy_model.weight = torch.nn.parameter.Parameter(W)
        return self.toy_model(x)


class MyModelParam(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.parameter.Parameter(torch.Tensor(2, 3).fill_(0.5))
        self.toy_model = ToyModel(weight=False)

    def forward(self, x):
        W = self.W
        # W = torch.exp(W)
        return self.toy_model(x, weight=W)


class ToyModel(torch.nn.Module):
    def __init__(self, weight=True):
        super().__init__()
        if weight:
            self.weight = torch.nn.parameter.Parameter(torch.Tensor(2, 3).fill_(0.5))
        else:
            self.register_parameter('weight', None)
        # self.func = torch.nn.parameter.Parameter(torch.Tensor(2, 2).fill_(0.1))

    def forward(self, x, weight=None):
        if weight is not None:
            w = weight
        else:
            w = self.weight
        # x = self.func * x
        x = torch.matmul(x, w)
        return x


x = torch.Tensor(1, 2).fill_(0.8)
y = torch.Tensor(1, 3).fill_(0.7)
print("################## Test 1: reset GCN.weight by nn.Parameter()##################")
model = MyModel()
print("toyModel weight before:")
print(model.toy_model.weight)
optim = torch.optim.SGD(model.parameters(), lr=1e-2)

# print("x: {}".format(x))
# print("y: {}".format(y))
prediction = model(x)
loss = (prediction - y).sum()
loss.backward()
optim.step()
print("toyModel weight after:")
print(model.toy_model.weight)

########################## Test 2
print("################## Test 2: pass weight as a parameter during forward##################")
model_param = MyModelParam()
print("model param weight before:")
print(model_param.W)
optim_param = torch.optim.SGD(model_param.parameters(), lr=1e-2)
x_param = torch.Tensor(1, 2).fill_(0.8)
y_param = torch.Tensor(1, 3).fill_(0.7)
# print("x_param: {}".format(x_param))
# print("y_param: {}".format(y_param))
prediction_param = model_param(x_param)
loss_param = (prediction_param - y_param).sum()
loss_param.backward()
optim_param.step()
print("model param weight after:")
print(model_param.W)