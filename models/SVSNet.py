import torch
from .sincnet import SincNet


class ResidualBlockW(torch.nn.Module):
    def __init__(self, dilation):
        super(ResidualBlockW, self).__init__()
        input_channel = 64
        skipped_channel = 64
        gate_channel = 128
        res_channel = 64
        self.input_conv = torch.nn.Conv1d(input_channel, gate_channel, kernel_size=3, dilation=dilation, padding=dilation)
        self.skipped_conv = torch.nn.Conv1d(gate_channel // 2, skipped_channel, kernel_size=1)
        self.res_conv = torch.nn.Conv1d(gate_channel // 2, res_channel, kernel_size=1)
        self.gc = gate_channel
    def forward(self, x):
        res = x
        gate_x = self.input_conv(x)
        xt, xs = torch.split(gate_x, self.gc // 2, dim=1)
        out = torch.tanh(xt) * torch.sigmoid(xs)
        s = self.skipped_conv(out)
        x = self.res_conv(out)
        x += res
        x *= 0.707
        return x, s


class WaveResNet(torch.nn.Module):
    def __init__(self):
        super(WaveResNet, self).__init__()
        input_channel = 64
        skipped_channel = 64
        layers = 7
        stacks = 1
        lps = layers // stacks
        self.first_conv = torch.nn.Conv1d(64, input_channel, kernel_size=1)
        self.conv = torch.nn.ModuleList([])
        for layer in range(layers):
            dilation = 2 ** (layer % lps)
            self.conv.append(ResidualBlockW(dilation=dilation))
        self.last_conv = torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Conv1d(skipped_channel, skipped_channel, kernel_size=1),
                torch.nn.ReLU(),
                torch.nn.Conv1d(skipped_channel, skipped_channel, kernel_size=1)
                )
    def forward(self, x):
        x = self.first_conv(x)
        s = 0
        for layer in self.conv:
            x, _s = layer(x)
            s += _s
        y = self.last_conv(s)
        return y


class model(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(model, self).__init__()
        self.sincnet = SincNet(N_cnn_lay=1, issinc=True, device=device)
#        self.ds = torch.nn.Sequential(
#                    torch.nn.Conv1d(1, 64, kernel_size=3, stride=3),
#                    torch.nn.MaxPool1d(kernel_size=3)
#                    )
        self.wavenet = torch.nn.ModuleList(
                        [WaveResNet() for i in range(4)])
        self.downsample = torch.nn.ModuleList(
                        [torch.nn.MaxPool1d(kernel_size=3) for i in range(4)])
        self.rnn = torch.nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True, num_layers=1)
        self.derive_model = torch.nn.Sequential(
                                 torch.nn.Linear(256, 128),
                                 torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                 torch.nn.Linear(128, 1)
                                 )
        for name, para in self.named_parameters():
            if 'bn' in name or 'pass' in  name:
                continue
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(para, gain=1.414)
                print('init '+name+' by xavier_uniform')
            if 'bias' in name:
                torch.nn.init.zeros_(para)
                print('init '+name+' by zero')
        self.to(device)
    def encode_frame_embedding(self, x):
#        y = [self.ds(_x.reshape(1, 1, -1)) for _x in x]
        y = [self.sincnet(_x) for _x in x]
        for i in range(4):
            y = [self.wavenet[i](_y) for _y in y]
            y = [self.downsample[i](_y) for _y in y]
        y = [self.rnn(_y.transpose(1,2))[0] for _y in y]
        return y
    def encode_frame_score(self, x):
        y = self.encode_frame_embedding(x)
        y = [self.derive_model(_y) for _y in y]
        return y
    def forward(self, x, label):
        y1 = self.encode_frame_embedding(x[:,0]) # BTF
        y2 = self.encode_frame_embedding(x[:,1]) # BTF
        atty1 = [torch.softmax(_y1.bmm(_y2.transpose(1,2)), dim=2).bmm(_y2) for _y1, _y2 in zip(y1, y2)] # BTF
        atty2 = [torch.softmax(_y2.bmm(_y1.transpose(1,2)), dim=2).bmm(_y1) for _y1, _y2 in zip(y1, y2)] # BTF
        y1 = torch.cat([(_atty.mean(dim=1) - _y1.mean(dim=1)).abs() for _atty, _y1 in zip(atty1, y1)], dim=0) # diff of mean embedding
        y2 = torch.cat([(_atty.mean(dim=1) - _y1.mean(dim=1)).abs() for _atty, _y1 in zip(atty2, y2)], dim=0) # diff of mean embedding
        y1 = self.derive_model(y1)
        y2 = self.derive_model(y2)
        y = (y1 + y2) / 2
        y = y.reshape(label.shape)
        return ((label-y)**2).mean(), y
    def predict(self, x, label):
        with torch.no_grad():
            return self(x, label)