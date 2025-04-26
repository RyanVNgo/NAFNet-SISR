
import torch

from .sisr_model import SISRModel, sisr_network_types
from .archs.PlainNet import PlainNet
from .archs.Baseline import Baseline
from .archs.NAFNet import NAFNet
from .losses import losses

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
]


def create_sisr_model(options):
    network_options = options.get('network_arch', {})

    net_type = network_options.get('type', 'PlainNet')
    c_in =  network_options.get('c_in', 3)
    width = network_options.get('width', 16)
    enc_blk_nums = network_options.get('enc_blk_nums', [1, 1, 1])
    dec_blk_nums = network_options.get('dec_blk_nums', [1, 1, 1])
    mid_blk_num = network_options.get('mid_blk_num', 1)
    intro_k = network_options.get('intro_k', 3)
    ending_k = network_options.get('ending_k', 3)
    block_opts = network_options.get('block', {})

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

    config = dict(
        network_arch = dict(
            type = net_type,
            width = width,
            enc_blk_nums = enc_blk_nums,
            mid_blk_num = mid_blk_num,
            dec_blk_nums = dec_blk_nums,
            intro_k = intro_k,
            ending_k = ending_k,
            block = dict(
                block_opts
            )
        )
    )

    return SISRModel(net, config, device)


def create_loss(type, options):
    weight = options.get('weight', 1.0)
    match type:
        case 'psnrloss':
            return losses.PSNRLoss(weight)
        case 'l1loss':
            return losses.L1Loss(weight)
        case 'mseloss':
            return losses.MSELoss(weight)
    return None


