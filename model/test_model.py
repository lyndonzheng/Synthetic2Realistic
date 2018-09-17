import torch
from torch.autograd import Variable
from .base_model import BaseModel
from . import network
from util import util
from collections import OrderedDict


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert (not opt.isTrain)
        BaseModel.initialize(self, opt)

        self.loss_names = []
        self.visual_names =['img_s', 'img_t', 'img_s2t', 'lab_t_g']
        # self.model_names = ['img2task', 's2t']
        self.model_names = ['img2task']

        # define the transform network
        self.net_s2t = network.define_G(opt.image_nc, opt.image_nc, opt.ngf, opt.transform_layers, opt.norm,
                                        opt.activation, opt.trans_model_type, opt.init_type, opt.drop_rate,
                                        False, opt.gpu_ids, opt.U_weight)
        # define the task network
        self.net_img2task = network.define_G(opt.image_nc, opt.label_nc, opt.ngf, opt.task_layers, opt.norm,
                                             opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
                                             False, opt.gpu_ids, opt.U_weight)

        self.load_networks(opt.which_epoch)

    def set_input(self, input):
        self.input = input
        self.img_source = input['img_source']
        self.img_target = input['img_target']

        if len(self.gpu_ids) > 0:
            self.img_source = self.img_source.cuda()
            self.img_target = self.img_target.cuda()

        self.image_paths = input['img_target_paths']

    def test(self):
        self.img_s = Variable(self.img_source)
        self.img_t = Variable(self.img_target)

        with torch.no_grad():
            self.img_s2t = self.net_s2t.forward(self.img_s)
            self.lab_t_g = self.net_img2task.forward(self.img_t)

    # save_results
    def save_results(self, visualizer, wed_page):
        img_source_paths = self.input['img_source_paths']
        img_target_paths = self.input['img_target_paths']

        for i in range(self.img_s.size(0)):
            img_source = util.tensor2im(self.img_s.data[i])
            img_target = util.tensor2im(self.img_t.data[i])
            img_source2target = util.tensor2im(self.img_s2t[-1].data[i])
            lab_fake_target = util.tensor2im(self.lab_t_g[-1].data[i])

            visuals = OrderedDict([('img_s', img_source), ('img_s2t', img_source2target)])
            print('process image ......%s' % img_source_paths[0])
            visualizer.save_images(wed_page, visuals, img_source_paths)
            img_source_paths.pop(0)

            visuals = OrderedDict([('img_t', img_target), ('lab_t_g', lab_fake_target)])
            print('process image ......%s' % img_target_paths[0])
            visualizer.save_images(wed_page, visuals, img_target_paths)
            img_target_paths.pop(0)