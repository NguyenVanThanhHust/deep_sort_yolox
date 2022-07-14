import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
try:
    from extract_feature.model import Net
except:
    from extract_feature.model import Net

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True

def get_extractor():
    num_classes = 752
    net = Net(num_classes=num_classes, reid=True)
    assert os.path.isfile("./weights/ckpt.pth"), "Error: no checkpoint file found!"
    print('Loading from ./weights/ckpt.pth')
    checkpoint = torch.load("./weights/ckpt.pth")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    net.to(device)
    return net