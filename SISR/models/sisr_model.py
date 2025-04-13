
import os
from os import path

from .archs.PlainNet import PlainNet
from .archs.Baseline import Baseline
from .archs.NAFNet import NAFNet


class SISRModel():
    network = None

    def __init__(self, net_type = 'PlainNet', c_in = 3):
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


    def run(self, input):
        if input.shape[-3] != self.c_in:
            return input
        output = self.network(input)
        return output


    def get_modules(self):
        modules = []
        for idx, module in enumerate(self.network.named_children()):
            modules.append(module)
        return modules


    def network_type(self):
        return self.network.__class__.__name__


    def input_channel_depth(self):
        return self.c_in



def sisr_network_types():
    network_dir = path.join(path.dirname(__file__), 'archs')
    network_types = [
        path.splitext(path.basename(f))[0] for f in os.listdir(network_dir) 
        if path.isfile(path.join(network_dir, f))
    ]
    return network_types

