from torch import nn
import torch
import torchvision
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

class USIDNet(nn.Module):
    def __init__(self):
        super(USIDNet, self).__init__()
        self.enc = Encode()
        self.dec = Decode()
    def load_weights(self, checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        self.enc.load_state_dict(state_dict['a'], strict=False)
        self.dec.load_state_dict(state_dict['b'], strict=False)
    def forward(self,x):
        return self.dec(self.enc(x))

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
        self.enc = Encoder()
    def forward(self,x):
        return self.enc(x)
class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
        self.dec_cont = Decoder()
    def forward(self,x):
        return self.dec_cont(x)

class Encoder(nn.Module):
    def __init__(self, n_downsample=2, input_dim=3, dim=64, norm='in', activ='relu', pad_type='reflect'):
        super(Encoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for _ in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for _ in range(4):
            self.model += [Bottle2neck(in_channels=dim, out_channels=dim, stride=1, stype='normal', norm='in'),]
        self.model = nn.Sequential(*self.model)
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self,n_upsample=2, dim=256, activ='relu', pad_type='reflect'):
        super(Decoder, self).__init__()
        self.model = []
        for _ in range(4):
            self.model += [Bottle2neck(in_channels=dim, out_channels=dim, stride=1, stype='normal', norm='in')]
        # upsampling blocks
        for _ in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        self.model += [CALayer(dim), PALayer(dim)]
        self.model += [Conv2dBlock(dim, 3, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
    def forward(self, x, x_a=None):
        output = self.model(x)
        return output

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Bottle2neck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal', norm='bn'):
        super(Bottle2neck, self).__init__()
        width = out_channels // 4
        if norm == 'bn':
            self.norm = nn.BatchNorm2d
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = self.norm(out_channels)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = self.norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                self.norm(out_channels),
            )

        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
        self.ca = CALayer(in_channels)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ca(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim , output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        self.norm_type = norm
        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm_type != 'wn' and self.norm != None:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class vgg_19(nn.Module):
    def __init__(self):
        super(vgg_19, self).__init__()
        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.feature_ext = nn.Sequential(*list(vgg_model.features.children())[:20])
    def forward(self, x):
        if x.size(1) == 1:
            x = torch.cat((x, x, x), 1)
        out = self.feature_ext(x)
        return out

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
