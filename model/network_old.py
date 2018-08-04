import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler


###############################################################################
# Functions
###############################################################################
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


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def _freeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True

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
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
        net.cuda()
    init_weights(net, init_type)
    return net


# Define the synthetic2real network
def define_TransNet(input_nc, output_nc, ngf=32, layers=3, norm='batch', use_dropout=False, init_type='xavier',
                    gpu_ids=[]):
    net = _ResnetGenerator(input_nc, output_nc, ngf, norm, use_dropout, layers, gpu_ids)

    return init_net(net, init_type, gpu_ids)


# Define the image2real network
def define_TaskNet(input_nc, output_nc, ngf=64, layers=4, norm='batch', use_dropout=False, init_type='xavier',
                   gpu_ids=[]):
    net = _UNet16(input_nc, output_nc, ngf, layers, norm, use_dropout, gpu_ids)

    return init_net(net, init_type, gpu_ids)


def define_D(input_nc, ndf=64, n_layers=3, num_D=1, norm='batch', init_type='xavier', gpu_ids=[]):
    net = MultiscaleDiscriminator(input_nc, ndf, n_layers, num_D, norm, gpu_ids)

    return init_net(net, init_type, gpu_ids)


def define_coderD(input_nc, n_layers=3, norm='batch', init_type='xavier', gpu_ids=[]):
    net = _codeDiscriminator(input_nc, n_layers, norm, gpu_ids)

    return init_net(net, init_type, gpu_ids)


###############################################################################
#   Basic model
###############################################################################

class GaussianNoiseLayer(nn.Module):
    def __init__(self, ):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable((torch.rand(x.size()).cuda(x.data.get_device()) - 0.5) / 5.0)
        return x + noise


class UpsampleBlock(nn.Module):
    def __init__(self, input_nc, up_scale, output_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(UpsampleBlock, self).__init__()
        model = [
            nn.Conv2d(input_nc, input_nc * up_scale ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(up_scale),
            nn.PReLU(),
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nn.PReLU()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class DecoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(DecoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nn.PReLU(),
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nn.PReLU()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(EncoderBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nn.PReLU(),
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, input):
        return self.block(input)


class EncoderBlock1(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(EncoderBlock1, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(input_nc, middle_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nn.PReLU(),
            nn.Conv2d(middle_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nn.PReLU(),
        )

    def forward(self, input):
        return self.block(input)


class OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(OutputBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(int(kernel_size / 2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, input):
        return self.block(input)


class Inception(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, width=1):
        super(Inception, self).__init__()

        self.width = width

        for i in range(width):
            layer = nn.Sequential(
                nn.ReflectionPad2d(i * 2 + 1),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=0, dilation=i * 2 + 1)
            )
            setattr(self, 'layer' + str(i), layer)  # print(classname)

        self.norm1 = norm_layer(output_nc * width)
        self.relu = nn.PReLU()
        self.branch1x1 = nn.Conv2d(output_nc * width, input_nc, kernel_size=3, padding=1)
        self.norm2 = norm_layer(input_nc)

    def forward(self, x):
        result = []
        for i in range(self.width):
            model = getattr(self, 'layer' + str(i))
            result.append(model(x))
        x = torch.cat(result, 1)
        x = self.relu(self.norm1(x))
        output = self.norm2(self.branch1x1(x))
        return output


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, input_nc, output_nc, width=1, sample=None, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_bias=False):
        super(ResnetBlock, self).__init__()
        self.sample = sample
        self.output_nc = output_nc

        self.conv_shortcut, self.conv_block = self.build_conv_block(input_nc, output_nc, width, sample, norm_layer,
                                                                    use_dropout, use_bias)
        self.relu = nn.PReLU()

    def build_conv_block(self, input_nc, output_nc, width, sample, norm_layer, use_dropout, use_bias):
        conv_block = []
        conv_shortcut = []

        if sample == 'down':
            conv_shortcut = [nn.ReflectionPad2d(1),
                             nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=2, padding=0, bias=use_bias),
                             norm_layer(output_nc)
                             ]
            conv_block += [nn.ReflectionPad2d(1),
                           nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=2, padding=0, bias=use_bias),
                           norm_layer(input_nc),
                           nn.PReLU(),
                           ]
        elif sample == 'up':
            conv_shortcut = [
                nn.ConvTranspose2d(input_nc, output_nc, kernel_size=1, stride=2, padding=0, output_padding=1,
                                   bias=use_bias),
                norm_layer(output_nc)
            ]
            conv_block += [
                nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(input_nc),
                nn.PReLU()
            ]

        elif sample == None:
            model = Inception(input_nc, output_nc, norm_layer, width)
            return nn.Sequential(*conv_shortcut), model

        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=0, bias=use_bias),
            norm_layer(output_nc)
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        return nn.Sequential(*conv_shortcut), nn.Sequential(*conv_block)

    def forward(self, x):
        if x.size(1) == self.output_nc and self.sample == None:
            out = x + self.conv_block(x)
        else:
            out = self.conv_shortcut(x) + self.conv_block(x)
        return self.relu(out)


class _ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm='batch', use_dropout=False, n_blocks=6, gpu_ids=[],
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(_ResnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(norm_type=norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 1
        mult = 1
        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (i + 1), 2)
            model += [nn.Conv2d(ngf * mult_prev, ngf * mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.PReLU()]

        mult = min(2 ** n_downsampling, 2)
        for i in range(n_blocks):
            model += [
                ResnetBlock(ngf * mult, ngf * mult, width=1, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]

        decoder = []
        mult = min(2 ** n_downsampling, 2)
        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (n_downsampling - i - 1), 2)
            model += [
                # UpsampleBlock(ngf * mult_prev, 2, ngf * mult, norm_layer, use_bias)
                DecoderBlock(ngf * mult_prev, ngf * mult_prev, ngf * mult, norm_layer, use_bias)
            ]

        model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Tanh()
            ]

        # self.encoder = nn.Sequential(*encoder)
        # self.decoder = nn.Sequential(*decoder)
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # feature = self.encoder(input)
        # result = [feature]
        # output = self.decoder(feature)
        # result.append(output)
        result = [self.model(input)]
        return result

###############################################################################
#   DC_GAN
###############################################################################

class _UNet16(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layers=4, norm='batch', use_dropout=False, gpu_ids=[]):
        super(_UNet16, self).__init__()

        self.gpu_ids = gpu_ids
        self.layers = layers
        norm_layer = get_norm_layer(norm_type=norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # H/2

        self.conv2 = EncoderBlock(ngf, ngf * 2, ngf * 2, norm_layer, use_bias)  # H/4 (128)
        self.conv3 = EncoderBlock(ngf * 2, ngf * 4, ngf * 4, norm_layer, use_bias)  # H/8 (256)
        self.conv4 = EncoderBlock(ngf * 4, ngf * 8, ngf * 8, norm_layer, use_bias)  # H/16 (512)

        for i in range(layers - 4):
            conv = EncoderBlock(ngf * 8, ngf * 8, ngf * 8, norm_layer, use_bias)
            setattr(self, 'down' + str(i), conv.model)

        center = []
        for i in range(7 - layers):
            center += [
                ResnetBlock(ngf * 8, ngf * 8, width=7 - layers, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)
            ]

        center += [DecoderBlock(ngf * 8, ngf * 8, ngf * 4, norm_layer, use_bias)]

        for i in range(layers - 4):
            # upconv = UpsampleBlock(ngf * (8 + 4), 2, ngf * 4, norm_layer, use_bias)
            upconv = DecoderBlock(ngf * (8 + 4), ngf * 8, ngf * 4, norm_layer, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = DecoderBlock(ngf * (4 + 4), ngf * 8, ngf * 2, norm_layer, use_bias)
        self.deconv3 = DecoderBlock(ngf * (2 + 2) + output_nc, ngf * 4, ngf, norm_layer, use_bias)
        self.deconv2 = DecoderBlock(ngf * (1 + 1) + output_nc, ngf * 2, int(ngf / 2), norm_layer, use_bias)
        self.deconv1 = OutputBlock(int(ngf / 2) + output_nc, output_nc, kernel_size=7, use_bias=use_bias)

        self.output4 = OutputBlock(ngf * (4 + 4), output_nc, kernel_size=3, use_bias=use_bias)
        self.output3 = OutputBlock(ngf * (2 + 2) + output_nc, output_nc, kernel_size=3, use_bias=use_bias)
        self.output2 = OutputBlock(ngf * (1 + 1) + output_nc, output_nc, kernel_size=3, use_bias=use_bias)

        self.center = nn.Sequential(*center)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2.forward(conv1)
        conv3 = self.conv3.forward(conv2)
        center_in = self.conv4.forward(conv3)

        middle = []
        for i in range(self.layers - 4):
            model = getattr(self, 'down' + str(i))
            center_in = model(center_in)
            middle.append(center_in)

        result = [center_in]

        center_out = self.center(center_in)

        for i in range(self.layers - 4):
            model = getattr(self, 'up' + str(i))
            center_out = model(torch.cat([center_out, middle[-i]], 1))

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


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3, norm='batch', gpu_ids=[]):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.gpu_ids = gpu_ids

        for i in range(num_D):
            netD = _DC_Discriminator(input_nc, ndf, n_layers, norm, gpu_ids)
            setattr(self, 'scale' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            model = getattr(self, 'scale' + str(i))
            output = model(input)
            result.append(output)
            if i != (self.num_D - 1):
                input = self.downsample(input)

        return result


class _DC_Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', gpu_ids=[]):
        super(_DC_Discriminator, self).__init__()

        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(norm_type=norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=7, stride=4, padding=2, bias=use_bias),
            nn.PReLU(),
        ]

        n_layers = n_layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.PReLU(),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class _codeDiscriminator(nn.Module):
    def __init__(self, input_nc, n_layers=2, norm='batch', gpu_ids=[]):
        super(_codeDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(norm_type=norm)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        model = [
            nn.Linear(input_nc * 12 * 40, input_nc),
            nn.PReLU(),
            nn.Linear(input_nc, input_nc),
            nn.PReLU()
        ]

        for i in range(1, n_layers - 1):
            model += [
                nn.Linear(input_nc, input_nc),
                nn.PReLU()
            ]

        model += [nn.Linear(input_nc, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        result = []
        input = input.view(-1, 512 * 12 * 40)
        output = self.model(input)
        result.append(output)
        return result