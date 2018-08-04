import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from torch.optim import lr_scheduler


######################################################################################
# Functions
######################################################################################
def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+1+opt.epoch_count-opt.niter) / float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params / 1e6))


def init_net(net, init_type='normal', gpu_ids=[]):

    print_network(net)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net = torch.nn.DataParallel(net, gpu_ids)
        net.cuda()
    init_weights(net, init_type)
    return net


def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def __unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

# define the generator(transform, task) network
def define_G(input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', model_type='UNet',
                    init_type='xavier', drop_rate=0, add_noise=False, gpu_ids=[], weight=0.1):

    if model_type == 'ResNet':
        net = _ResGenerator(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids)
    elif model_type == 'UNet':
        net = _UNetGenerator(input_nc, output_nc, ngf, layers, norm, activation, drop_rate, add_noise, gpu_ids, weight)
        # net = _PreUNet16(input_nc, output_nc, ngf, layers, True, norm, activation, drop_rate, gpu_ids)
    else:
        raise NotImplementedError('model type [%s] is not implemented', model_type)

    return init_net(net, init_type, gpu_ids)

# define the discriminator network
def define_D(input_nc, ndf = 64, n_layers = 3, num_D = 1, norm = 'batch', activation = 'PReLU', init_type='xavier', gpu_ids = []):

    net = _MultiscaleDiscriminator(input_nc, ndf, n_layers, num_D, norm, activation, gpu_ids)

    return init_net(net, init_type, gpu_ids)


# define the feature discriminator network
def define_featureD(input_nc, n_layers=2, norm='batch', activation='PReLU', init_type='xavier', gpu_ids=[]):

    net = _FeatureDiscriminator(input_nc, n_layers, norm, activation, gpu_ids)

    return init_net(net, init_type, gpu_ids)

######################################################################################
# Basic Operation
######################################################################################

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = torch.nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = torch.nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = torch.nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable((torch.randn(x.size()).cuda(x.data.get_device()) - 0.5) / 10.0)
        return x+noise

class _InceptionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), width=1, drop_rate=0, use_bias=False):
        super(_InceptionBlock, self).__init__()

        self.width = width
        self.drop_rate = drop_rate

        for i in range(width):
            layer = nn.Sequential(
                nn.ReflectionPad2d(i*2+1),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=0, dilation=i*2+1, bias=use_bias)
            )
            setattr(self, 'layer'+str(i), layer)

        self.norm1 = norm_layer(output_nc * width)
        self.norm2 = norm_layer(output_nc)
        self.nonlinearity = nonlinearity
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc * width, output_nc, kernel_size=3, padding=0, bias=use_bias)
        )


    def forward(self, x):
        result = []
        for i in range(self.width):
            layer = getattr(self, 'layer'+str(i))
            result.append(layer(x))
        output = torch.cat(result, 1)
        output = self.nonlinearity(self.norm1(output))
        output = self.norm2(self.branch1x1(output))
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate, training=self.training)

        return self.nonlinearity(output+x)


class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_EncoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _DownBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DownBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity,
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _ShuffleUpBlock(nn.Module):
    def __init__(self, input_nc, up_scale, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_ShuffleUpBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, input_nc*up_scale**2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.PixelShuffle(up_scale),
            nonlinearity,
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(_OutputBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


######################################################################################
# Network structure
######################################################################################

class _ResGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[]):
        super(_ResGenerator, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        ]

        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (i+1), 2)
            encoder += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(ngf*mult_prev, ngf*mult, kernel_size=3, stride=2, padding=0, bias=use_bias),
                norm_layer(ngf*mult),
                nonlinearity
            ]

        mult = min(2 ** n_downsampling, 2)
        for i in range(n_blocks-n_downsampling):
            encoder +=[
                _InceptionBlock(ngf*mult, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, width=1,
                                drop_rate=drop_rate, use_bias=use_bias)
            ]

        decoder = []
        if add_noise:
            decoder += [GaussianNoiseLayer()]

        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (n_downsampling - i -1), 2)
            decoder +=[
                _DecoderUpBlock(ngf*mult_prev, ngf*mult_prev, ngf*mult, norm_layer, nonlinearity, use_bias),
            ]

        decoder +=[
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        feature = self.encoder(input)
        result = [feature]
        output = self.decoder(feature)
        result.append(output)
        return result


class _PreUNet16(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=5, pretrained=False,  norm ='batch', activation='PReLu',
                 drop_rate=0, gpu_ids=[]):
        super(_PreUNet16, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = models.vgg16(pretrained=pretrained).features

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(encoder[0], self.relu, encoder[2], self.relu)
        self.conv2 = nn.Sequential(encoder[5], self.relu, encoder[7], self.relu)
        self.conv3 = nn.Sequential(encoder[10], self.relu, encoder[12], self.relu, encoder[14], self.relu)
        self.conv4 = nn.Sequential(encoder[17], self.relu, encoder[19], self.relu, encoder[21], self.relu)

        for i in range(layers - 4):
            conv = _EncoderBlock(ngf * 8, ngf * 8, ngf * 8, norm_layer, nonlinearity, use_bias)
            setattr(self, 'down' + str(i), conv.model)

        center = []
        for i in range(7 - layers):
            center += [
                _InceptionBlock(ngf * 8, ngf * 8, norm_layer, nonlinearity, 7 - layers, drop_rate, use_bias)
            ]

        center += [_DecoderUpBlock(ngf * 8, ngf * 8, ngf * 4, norm_layer, nonlinearity, use_bias)]

        for i in range(layers - 4):
            upconv = _DecoderUpBlock(ngf * (8 + 4), ngf * 8, ngf * 4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(ngf * (4 + 4), ngf * 8, ngf * 2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(ngf * (2 + 2) + output_nc, ngf * 4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(ngf * (1 + 1) + output_nc, ngf * 2, int(ngf / 2), norm_layer, nonlinearity, use_bias)

        self.deconv1 = _OutputBlock(int(ngf / 2) + output_nc, output_nc, kernel_size=7, use_bias=use_bias)

        self.output4 = _OutputBlock(ngf * (4 + 4), output_nc, kernel_size=3, use_bias=use_bias)
        self.output3 = _OutputBlock(ngf * (2 + 2) + output_nc, output_nc, kernel_size=3, use_bias=use_bias)
        self.output2 = _OutputBlock(ngf * (1 + 1) + output_nc, output_nc, kernel_size=3, use_bias=use_bias)

        self.center = nn.Sequential(*center)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input):
        conv1 = self.pool(self.conv1(input))
        conv2 = self.pool(self.conv2(conv1))
        conv3 = self.pool(self.conv3(conv2))
        center_in = self.pool(self.conv4(conv3))

        middle = [center_in]
        for i in range(self.layers - 4):
            model = getattr(self, 'down' + str(i))
            center_in = self.pool(model(center_in))
            middle.append(center_in)

        result = [center_in]

        center_out = self.center(center_in)

        for i in range(self.layers - 4):
            model = getattr(self, 'up' + str(i))
            center_out = model(torch.cat([center_out, middle[self.layers - 4 - i]], 1))

        deconv4 = self.deconv4.forward(torch.cat([center_out, conv3 * 0.1], 1))
        output4 = self.output4.forward(torch.cat([center_out, conv3 * 0.1], 1))
        result.append(output4)
        deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2 * 0.05, self.upsample(output4)], 1))
        output3 = self.output3.forward(torch.cat([deconv4, conv2 * 0.05, self.upsample(output4)], 1))
        result.append(output3)
        deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1 * 0.01, self.upsample(output3)], 1))
        output2 = self.output2.forward(torch.cat([deconv3, conv1 * 0.01, self.upsample(output3)], 1))
        result.append(output2)

        output1 = self.deconv1.forward(torch.cat([deconv2, self.upsample(output2)], 1))
        result.append(output1)

        return result

class _UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[],
                 weight=0.1):
        super(_UNetGenerator, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        )
        self.conv2 = _EncoderBlock(ngf, ngf*2, ngf*2, norm_layer, nonlinearity, use_bias)
        self.conv3 = _EncoderBlock(ngf*2, ngf*4, ngf*4, norm_layer, nonlinearity, use_bias)
        self.conv4 = _EncoderBlock(ngf*4, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)

        for i in range(layers-4):
            conv = _EncoderBlock(ngf*8, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)
            setattr(self, 'down'+str(i), conv.model)

        center=[]
        for i in range(7-layers):
            center +=[
                _InceptionBlock(ngf*8, ngf*8, norm_layer, nonlinearity, 7-layers, drop_rate, use_bias)
            ]

        center += [
        _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
        ]
        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        for i in range(layers-4):
            upconv = _DecoderUpBlock(ngf*(8+4), ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(ngf*(2+2)+output_nc, ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(ngf*(1+1)+output_nc, ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.output4 = _OutputBlock(ngf*(4+4), output_nc, 3, use_bias)
        self.output3 = _OutputBlock(ngf*(2+2)+output_nc, output_nc, 3, use_bias)
        self.output2 = _OutputBlock(ngf*(1+1)+output_nc, output_nc, 3, use_bias)
        self.output1 = _OutputBlock(int(ngf/2)+output_nc, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        conv1 = self.pool(self.conv1(input))
        conv2 = self.pool(self.conv2.forward(conv1))
        conv3 = self.pool(self.conv3.forward(conv2))
        center_in = self.pool(self.conv4.forward(conv3))

        middle = [center_in]
        for i in range(self.layers-4):
            model = getattr(self, 'down'+str(i))
            center_in = self.pool(model.forward(center_in))
            middle.append(center_in)
        center_out = self.center.forward(center_in)
        result = [center_in]

        for i in range(self.layers-4):
            model = getattr(self, 'up'+str(i))
            center_out = model.forward(torch.cat([center_out, middle[self.layers-5-i]], 1))

        deconv4 = self.deconv4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        output4 = self.output4.forward(torch.cat([center_out, conv3 * self.weight], 1))
        result.append(output4)
        deconv3 = self.deconv3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        output3 = self.output3.forward(torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        result.append(output3)
        deconv2 = self.deconv2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        output2 = self.output2.forward(torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        result.append(output2)
        output1 = self.output1.forward(torch.cat([deconv2, self.upsample(output2)], 1))
        result.append(output1)

        return result


class _MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=1, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.gpu_ids = gpu_ids

        for i in range(num_D):
            netD = _Discriminator(input_nc, ndf, n_layers, norm, activation, gpu_ids)
            setattr(self, 'scale'+str(i), netD)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            netD = getattr(self, 'scale'+str(i))
            output = netD.forward(input)
            result.append(output)
            if i != (self.num_D-1):
                input = self.downsample(input)
        return result


class _Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_Discriminator, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            nonlinearity,
        ]

        nf_mult=1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            model += [
                SpectralNorm(nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias)),
                norm_layer(ndf*nf_mult),
                nonlinearity,
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias)),
            norm_layer(ndf * 8),
            nonlinearity,
            SpectralNorm(nn.Conv2d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class _FeatureDiscriminator(nn.Module):
    def __init__(self, input_nc, n_layers=2, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_FeatureDiscriminator, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            SpectralNorm(nn.Linear(input_nc * 40 * 12, input_nc)),
            nonlinearity,
        ]

        for i in range(1, n_layers):
            model +=[
                SpectralNorm(nn.Linear(input_nc, input_nc)),
                nonlinearity
            ]

        model +=[SpectralNorm(nn.Linear(input_nc, 1))]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        result = []
        input = input.view(-1, 512 * 40 * 12)
        output = self.model(input)
        result.append(output)
        return result