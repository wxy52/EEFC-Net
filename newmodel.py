import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from omegaconf import DictConfig
from base import BaseModel
from einops import rearrange
from decode import DEC
import torch
import torch.nn as nn
from einops import rearrange


class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

        self.unfold = nn.Unfold(kernel_size=self.kernel_size,padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2))

        self.get_weights = nn.Sequential(
            nn.Conv2d(in_channel * (self.kernel_size[0] * self.kernel_size[1]),
                      in_channel * (self.kernel_size[0] * self.kernel_size[1]),
                      kernel_size=1, groups=in_channel),
            nn.BatchNorm2d(in_channel * (self.kernel_size[0] * self.kernel_size[1]))
        )

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=self.kernel_size,
                              padding=0, stride=self.kernel_size)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        unfold_feature = self.unfold(x)
        print(unfold_feature.shape)
        x = unfold_feature
        data = unfold_feature.unsqueeze(-1)


        weight = self.get_weights(data).view(b, c, self.kernel_size[0] * self.kernel_size[1], h, w) \
            .permute(0, 1, 3, 4, 2).softmax(-1)


        weight_out = rearrange(weight, 'b c h w (n1 n2) -> b c (h n1) (w n2)',
                               n1=self.kernel_size[0], n2=self.kernel_size[1])

        receptive_field_data = rearrange(x, 'b (c n1) l -> b c n1 l', n1=self.kernel_size[0] * self.kernel_size[1]) \
            .permute(0, 1, 3, 2).reshape(b, c, h, w, self.kernel_size[0] * self.kernel_size[1])

        data_out = rearrange(receptive_field_data, 'b c h w (n1 n2) -> b c (h n1) (w n2)',
                             n1=self.kernel_size[0], n2=self.kernel_size[1])

        conv_data = data_out * weight_out
        conv_out = self.conv(conv_data)

        return self.act(self.bn(conv_out))


class ETEBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
    #in_planes：输入的通道数。planes：输出的通道数。roi_num：卷积核的大小参数。bias：是否使用偏置项。
        super().__init__()
        self.d = roi_num
        self.cnn1 = torch.nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = torch.nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)

class rfaE2EBlock(torch.nn.Module):
    '''E2Eblock.'''

    def __init__(self, in_planes, planes, roi_num, bias=True):
    #in_planes：输入的通道数。planes：输出的通道数。roi_num：卷积核的大小参数。bias：是否使用偏置项。
        super().__init__()
        self.d = roi_num
        self.cnn1 = RFAConv(in_planes, planes, (1, self.d))
        self.cnn2 = RFAConv(in_planes, planes, (self.d, 1))

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a]*self.d, 3)+torch.cat([b]*self.d, 2)

class Newmodel(BaseModel):
    def __init__(self):
        super().__init__()
        self.node_num = 200
        self.e2econv1 = ETEBlock(1, 32, self.node_num, bias=True)
        self.e2econv2 = ETEBlock(32, 64, self.node_num, bias=True)
        self.E2N = torch.nn.Conv2d(64, 1, (1, self.node_num))
        self.N2G = torch.nn.Conv2d(1, 256, (self.node_num, 1))

        self.attention_list = nn.ModuleList()
        for _ in range(2):
            self.attention_list.append(
                TransformerEncoderLayer(d_model=self.node_num, nhead=2, dim_feedforward=1024,
                                        batch_first=True)
            )
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.node_num, 8),
            nn.LeakyReLU()
        )
        final_dim = 256

        self.fc = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )
        self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                       orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def forward(self,
                node_feature: torch.tensor):
        for atten in self.attention_list:
            out = atten(node_feature)
        out = out.unsqueeze(dim=1) #(32,1,200,200)
        out = F.leaky_relu(self.e2econv1(out), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33) #torch.Size([16, 64, 200, 200])
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)  # torch.Size([16, 1, 200, 1])
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)  # torch.Size([16, 256, 1, 1])
        out = out.view(out.size(0), -1)  # torch.Size([16, 256])

        return self.fc(out)

if __name__ == "__main__":
    example = torch.rand(32,200,200)
    net = Newmodel()
    print(net(example))