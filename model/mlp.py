import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)
    
    # WTF?
    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
    
if __name__ == "__main__":
    # test
    from torchsummary import summary
    hidden_list = [16, 8, 8, 4]
    model = MLP(16, 3, hidden_list).cuda()
    summary(model, input_size = (3, 16, 16), batch_size=1)