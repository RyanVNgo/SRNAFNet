
import os
from os import path

import torch
import yaml


class SISRModel():

    def __init__(self, net, config, device='cpu'):
        self.model_save_path = None
        self.config_save_path = None
        self.config = config 
        self.net = net
        self.device = device
        self.net.to(device)

    def set_eval(self):
        self.net.eval()

    def set_train(self):
        self.net.train(True)

    def predict(self, input):
        if input.shape[-3] != self.net.c_in:
            return input
        return self.net(input)
    
    def get_parameters(self):
        return self.net.parameters()

    def get_modules(self):
        modules = []
        for idx, module in enumerate(self.net.named_children()):
            modules.append(module)
        return modules

    def to_device(self, device):
        self.net.to(device)

    def curr_device(self):
        return self.device

    def network_type(self):
        return self.net.__class__.__name__

    def input_channel_depth(self):
        return self.net.c_in

    def set_model_save_path(self, path):
        self.model_save_path = path

    def set_config_save_path(self, path):
        self.config_save_path = path

    def load_model(self, path):
        self.net.load_state_dict(
            torch.load(
                path, 
                weights_only=True, 
                map_location=self.device
            ),
            strict=False
        )

    def save_model(self, path=None):
        if path is not None:
            torch.save(self.net.state_dict(), path)
        elif self.model_save_path is not None:
            torch.save(self.net.state_dict(), self.model_save_path)

    def save_config(self, path=None):
        if path is not None:
            with open(path, 'w') as net_config_file:
                yaml.dump(self.config, net_config_file, default_flow_style=False)
        elif self.config_save_path is not None:
            with open(self.config_save_path, 'w') as net_config_file:
                yaml.dump(self.config, net_config_file, default_flow_style=False)

def sisr_network_types():
    network_dir = path.join(path.dirname(__file__), 'archs')
    network_types = [
        path.splitext(path.basename(f))[0] for f in os.listdir(network_dir) 
        if path.isfile(path.join(network_dir, f))
    ]
    return network_types

