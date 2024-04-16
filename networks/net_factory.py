from networks.unet import UNet, UNet_2d, UNet_2d_ECL
from networks.VNet import VNet, VNet_3d_ECL


def net_factory(net_type="unet", in_chns=1, class_num=2, mode="train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "VNet" and mode == "train" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet" and mode == "test" and tsne == 0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net


def BCP_net(in_chns=1, class_num=2, ema=False):
    net = UNet_2d(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def BCP_net_ECL(in_chns=1, class_num=2, ema=False):
    net = UNet_2d_ECL(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

def VNet_ECL(in_chns=1, class_num=2):
    net = VNet_3d_ECL(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    return net


def VNet_3d(in_chns=1, class_num=2):
    net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net
