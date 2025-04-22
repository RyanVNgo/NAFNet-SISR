
from .sisr_model import SISRModel, sisr_network_types
from .archs.PlainNet import PlainNet
from .archs.Baseline import Baseline
from .archs.NAFNet import NAFNet

__all__ = [
    'create_sisr_model'

    # sisr_model.py
    'SISRModel',
    'sisr_network_types'

    # archs
    'PlainNet'
    'Baseline'
    'NAFNet'
]


def create_sisr_model(options):
    network_options = options.get('network_arch', None)

    if network_options is None:
        return default_sisr_net()

    net_type = network_options.get('type', 'PlainNet')
    c_in =  network_options.get('c_in', 3)
    width = network_options.get('width', 16)
    enc_blk_nums = network_options.get('enc_blk_nums', [1, 1, 1])
    dec_blk_nums = network_options.get('dec_blk_nums', [1, 1, 1])
    mid_blk_num = network_options.get('mid_blk_num', 1)

    device = options.get('device', 'cpu')

    net = None
    match net_type:
        case 'PlainNet':
            net = PlainNet(
                c_in=c_in,
                width=width,
                mid_blk_num=mid_blk_num,
                enc_blk_nums=enc_blk_nums,
                dec_blk_nums=dec_blk_nums
            )
        case 'Baseline':
            net = Baseline(
                c_in=c_in,
                width=width,
                mid_blk_num=mid_blk_num,
                enc_blk_nums=enc_blk_nums,
                dec_blk_nums=dec_blk_nums
            )
        case 'NAFNet':
            net = NAFNet(
                c_in=c_in,
                width=width,
                mid_blk_num=mid_blk_num,
                enc_blk_nums=enc_blk_nums,
                dec_blk_nums=dec_blk_nums
            )

    return SISRModel(net, device)


def default_sisr_net():
    model = PlainNet (
        c_in=3,
        width=16,
        mid_blk_num=1,
        enc_blk_nums=[1, 1, 1],
        dec_blk_nums=[1, 1, 1]
    )
    return model


