
import os
from os import path

import torch

from .archs.PlainNet import PlainNet
from .archs.Baseline import Baseline
from .archs.NAFNet import NAFNet


class SISRModel():
    network = None

    def __init__(self, net_type=None, c_in=3, device=None):
        self.c_in = c_in
        match net_type:
            case 'PlainNet':
                self.network = PlainNet(c_in)
            case 'Baseline':
                self.network = Baseline(c_in)
            case 'NAFNet':
                self.network = NAFNet(c_in)
            case _:
                self.network = PlainNet(c_in)

        if device is not None:
            try:
                self.device = torch.device(device)
            except:
                pass
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.network.to(self.device)

    def set_eval(self):
        self.network.eval()

    def set_train(self):
        self.network.train(True)

    def predict(self, input):
        if input.shape[-3] != self.c_in:
            return input
        return self.network(input)
    
    def get_parameters(self):
        return self.network.parameters()

    def get_modules(self):
        modules = []
        for idx, module in enumerate(self.network.named_children()):
            modules.append(module)
        return modules

    def curr_device(self):
        return self.device

    def network_type(self):
        return self.network.__class__.__name__

    def input_channel_depth(self):
        return self.c_in

    def save_model(self, path):
        torch.save(self.network.state_dict(), path)

    def load_model(self, path):
        self.network.load_state_dict(torch.load(path, weights_only=True))


def sisr_network_types():
    network_dir = path.join(path.dirname(__file__), 'archs')
    network_types = [
        path.splitext(path.basename(f))[0] for f in os.listdir(network_dir) 
        if path.isfile(path.join(network_dir, f))
    ]
    return network_types

