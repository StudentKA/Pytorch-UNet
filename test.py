import torch
from optparse import OptionParser

from eval import eval_net
from unet import UNet
from utils import get_ids, get_imgs_and_masks, batch

dir_img = '/content/Pytorch-UNet/result/test/'
dir_mask = '/content/Pytorch-UNet/result/test_y/'

def get_result(net, gpu=False):
    ids = get_ids(dir_img)

    val = get_imgs_and_masks(ids, dir_img, dir_mask, 1.0)

    val_dice = eval_net(net, val, gpu)
    print('Validation Dice Coeff: {}'.format(val_dice))

def get_args():
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)

    if args.load:
        if args.gpu:
            net.load_state_dict(torch.load(args.load))
        else:
            net.load_state_dict(torch.load(args.load, map_location='cpu'))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()

    get_result(net, args.gpu)
