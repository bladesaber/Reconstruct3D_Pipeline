from tensorboardX import SummaryWriter
import visdom
import numpy as np

class Logger_Tensorboard(object):
    def __init__(self, log_dir):
        self.tb_writer = SummaryWriter(logdir=log_dir)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        for tag, value in tag_value_pairs:
            self.tb_writer.add_scalar(tag, value, global_step=step)

    def scalar_summary(self, tag, value, step):
        self.tb_writer.add_scalar(tag, value, global_step=step)

    def dict_of_scalars_summary(self, tag_dict, step):
        for tag in tag_dict:
            self.tb_writer.add_scalar(tag, tag_dict[tag], global_step=step)

    def histogram_summary(self, tag, value):
        self.tb_writer.add_histogram(tag, value, 0)

class Logger_Visdom(object):
    def __init__(self):
        self.vis = visdom.Visdom()

    def log_img(self, image, name):
        image = np.transpose(image, (2, 0, 1))
        self.vis.image(image, win=name)

class Metric_Recorder(object):
    def __init__(self):
        self.record_dict = {}

    def add_scalar_tensor(self, name, scalar):
        scalar_float = scalar.cpu().detach().item()

        if name not in self.record_dict.keys():
            self.record_dict[name] = []
        self.record_dict[name].append(scalar_float)
        return scalar_float

    def clear(self):
        for name in self.record_dict.keys():
            self.record_dict[name].clear()

    def compute_mean(self):
        result = {}
        for name in self.record_dict.keys():
            result[name] = np.mean(self.record_dict[name])
        return result
