
import os
import torch

from .sisr_model import SISRModel, sisr_network_types
from .archs.PlainNet import PlainNet
from .archs.Baseline import Baseline
from .archs.NAFNet import NAFNet
from .archs.SRNAFNet import SRNAFNet, nafnet_weight_init
from .archs.discriminator import Discriminator
from .losses import losses

import utils


__all__ = [
    'create_sisr_model'
    'create_loss'

    # sisr_model.py
    'SISRModel',
    'sisr_network_types'

    # archs
    'PlainNet'
    'Baseline'
    'NAFNet'
    'SRNAFNet'
    'Discriminator'
]


def create_sisr_model(options):
    network_options = options.get('network_arch', {})
    
    load_path = options.get('load_path', None)
    ignore_yaml = options.get('ignore_yaml', False)

    yaml_path = None
    if load_path is not None:
        load_path = os.path.abspath(load_path)
        if os.path.exists(load_path):
            dir = os.path.dirname(load_path)
            filename = os.path.basename(load_path)
            filename = os.path.splitext(filename)[0] + '.yaml'
            yaml_path = os.path.join(dir, filename)

    if yaml_path is not None and ignore_yaml == False:
        print('Network loaded')
        network_options = utils.parse_options(yaml_path).get('network_arch', {})

    net_type = network_options.get('type', 'PlainNet')
    c_in =  network_options.get('c_in', 3)
    width = network_options.get('width', 16)
    enc_blk_nums = network_options.get('enc_blk_nums', [1, 1, 1])
    dec_blk_nums = network_options.get('dec_blk_nums', [1, 1, 1])
    mid_blk_num = network_options.get('mid_blk_num', 1)
    intro_k = network_options.get('intro_k', 3)
    ending_k = network_options.get('ending_k', 3)
    block_opts = network_options.get('block', {})

    sfe_k_nums = network_options.get('sfe_k_nums', [3, 5, 7]) 
    dfe_count = network_options.get('dfe_count', 1) 
    dfe_k = network_options.get('dfe_k', 3) 
    ufe_count = network_options.get('ufe_count', 1) 
    ufe_k = network_options.get('ufe_k', 3) 

    device = options.get('device', 'cpu')
    if not torch.cuda.is_available():
        device = torch.device('cpu')

    net = None
    match net_type:
        case 'PlainNet':
            net = PlainNet(
                c_in=c_in,
                width=width,
                mid_blk_num=mid_blk_num,
                enc_blk_nums=enc_blk_nums,
                dec_blk_nums=dec_blk_nums,
                intro_k=intro_k,
                ending_k=ending_k,
                block_opts=block_opts
            )
        case 'Baseline':
            net = Baseline(
                c_in=c_in,
                width=width,
                mid_blk_num=mid_blk_num,
                enc_blk_nums=enc_blk_nums,
                dec_blk_nums=dec_blk_nums,
                intro_k=intro_k,
                ending_k=ending_k,
                block_opts=block_opts
            )
        case 'NAFNet':
            net = NAFNet(
                c_in=c_in,
                width=width,
                mid_blk_num=mid_blk_num,
                enc_blk_nums=enc_blk_nums,
                dec_blk_nums=dec_blk_nums,
                intro_k=intro_k,
                ending_k=ending_k,
                block_opts=block_opts
            )
        case 'SRNAFNet':
            net = SRNAFNet(
                c_in=c_in,
                width=width,
                sfe_k_nums=sfe_k_nums,
                dfe_count=dfe_count,
                dfe_k=dfe_k,
                ufe_count=ufe_count,
                ufe_k=ufe_k,
                intro_k=intro_k,
                ending_k=ending_k,
                block_opts=block_opts
            )

    config = dict(
        network_arch = dict(
            type = net_type,
            width = width,
            enc_blk_nums = enc_blk_nums,
            mid_blk_num = mid_blk_num,
            dec_blk_nums = dec_blk_nums,
            sfe_k_nums=sfe_k_nums,
            dfe_count=dfe_count,
            dfe_k=dfe_k,
            ufe_count=ufe_count,
            ufe_k=ufe_k,
            intro_k = intro_k,
            ending_k = ending_k,
            block = dict(
                block_opts
            )
        )
    )

    net.apply(lambda m: nafnet_weight_init(m, scale=0.1))
    model = SISRModel(net, config, device)
    if load_path is not None:
        model.load_model(load_path)
    return model


def create_loss(type, options, device):
    weight = options.get('weight', 1.0)
    match type:
        case 'psnrloss':
            return losses.PSNRLoss(weight)
        case 'l1loss':
            return losses.L1Loss(weight)
        case 'mseloss':
            return losses.MSELoss(weight)
        case 'huberloss':
            return losses.HuberLoss(weight)
        case 'vggloss':
            return losses.VGGLoss(conv_index='22', weight=weight, device=device)
    return None


