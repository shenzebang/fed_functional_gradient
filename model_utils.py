from model import convnet, mlp, resnet
import torch.nn as nn
import torch
def make_model(args, n_classes, n_channels, device):
    dense_hidden_size = tuple([int(a) for a in args.dense_hid_dims.split("-")])
    conv_hidden_size = tuple([int(a) for a in args.conv_hid_dims.split("-")])

    if args.model == "convnet":
        model = convnet.LeNet5(n_classes, n_channels, conv_hidden_size, dense_hidden_size, device)
    elif args.model == "mlp":
        model = mlp.MLP(n_classes, dense_hidden_size, device)
    elif args.model == "resnet":
        model = resnet.resnet20().to(device)
    else:
        raise NotImplementedError

    return model

class FunctionEnsemble(nn.Module):
    def __init__(self):
        super(FunctionEnsemble, self).__init__()
        self.function_list = []
        self.weight_list = []

    def forward(self, x):
        if len(self.function_list) != len(self.weight_list):
            raise RuntimeError
        if len(self.function_list) == 0:
            return 0.
        y = torch.sum(
                torch.stack(
                    [weight * function(x)
                     for function, weight in zip(self.function_list, self.weight_list)]
                ),
                dim = 0
            )

        return y

    def add_function(self, f, weight):
        self.function_list.append(f)
        self.weight_list.append(weight)

    def add_ensemble(self, ensemble):
        self.function_list = self.function_list + ensemble.function_list
        self.weight_list = self.weight_list + ensemble.weight_list

    def rescale_weights(self, factor):
        self.weight_list = [weight * factor for weight in self.weight_list]
