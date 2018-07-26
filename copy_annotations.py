import os, tqdm, random, pickle

import torch
import torchvision

from torch.autograd import Variable
from torchvision.transforms import CenterCrop, ToTensor, Compose, Lambda, Resize
from torchvision.datasets import coco
from torch.nn.functional import binary_cross_entropy, relu, nll_loss
from torch.nn import Embedding
from torch.optim import Adam

import nltk

from argparse import ArgumentParser

from collections import defaultdict, Counter, OrderedDict

import util, models

from util import PAD, SOS, EOS, UNK, EXTRA_SYMBOLS

from tensorboardX import SummaryWriter

def go(arg):

    transform = Compose([
        ToTensor()])

    imdir = arg.data_dir + os.sep + 'val2017'
    anfile = arg.data_dir + os.sep + 'annotations' + os.sep + 'captions_val2017.json'

    coco_data = coco.CocoCaptions(root=imdir, annFile=anfile, transform=transform)

    ## Make a dictionary

    with open('coco.valannotations.txt', 'w') as file:
        print('Copying annotations.') # Why is this so slow?

        dist = Counter()
        for i in tqdm.trange(len(coco_data)):
            for caption in coco_data[i][1]:
                file.write(caption + '\n')

    print('Done.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)