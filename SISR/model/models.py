
from .archs.PlainNet import PlainNet


class BaseModel():
    model = None

    def __init__(self, c_in, depth = 3, dw_expand = 1, ffn_expand = 2, dropout = 0.0):
        self.model = PlainNet(c_in, depth, dw_expand, ffn_expand, dropout)
        self.input_channels = c_in 


    def model_run(self, input):
        if input.shape[-3] != self.input_channels:
            return input
        output = self.model(input)
        return output


    def get_input_channels(self):
        return self.input_channels 


    def get_modules(self):
        modules = []
        for idx, module in enumerate(self.model.named_children()):
            modules.append(module)
        return modules


