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

from tensorboardX import SummaryWriter

EXTRA_SYMBOLS = ['.pad', '.sos', '.eos', '.unk']

def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    transform = Compose([
        Lambda(lambda x: CenterCrop(min(x.size))(x)),
        Resize(size=(arg.img_size, arg.img_size)),
        ToTensor()])

    imdir = arg.data_dir + os.sep + 'val2017'
    anfile = arg.data_dir + os.sep + 'annotations' + os.sep + 'captions_val2017.json'

    coco_data = coco.CocoCaptions(root=imdir, annFile=anfile, transform=transform)

    ## Make a dictionary

    util.ensure(arg.cache_dir)
    if os.path.isfile(arg.cache_dir + os.sep + 'i2w.pkl'):
        with open(arg.cache_dir + os.sep + 'i2w.pkl', 'rb') as file:
            i2w = pickle.load(file)
        with open(arg.cache_dir + os.sep + 'w2i.pkl', 'rb') as file:
            w2i = pickle.load(file)
        print('Word indices loaded.')
    else:
        print('Creating word indices') # Why is this so slow?

        dist = Counter()
        for i in tqdm.trange(len(coco_data)):
            for caption in coco_data[i][1]:
                dist.update(util.tokenize(caption))

        vocab = dist.most_common(arg.max_vocab - len(EXTRA_SYMBOLS))

        i2w = EXTRA_SYMBOLS + [w[0] for w in vocab]
        w2i = {word:ix for ix, word in enumerate(i2w)}

        with open(arg.cache_dir + os.sep + 'i2w.pkl', 'wb') as file:
            pickle.dump(i2w, file)
        with open(arg.cache_dir + os.sep + 'w2i.pkl', 'wb') as file:
            pickle.dump(w2i, file)

    vocab_size = len(i2w)
    print('vocabulary size', vocab_size)
    print('top 100 words:', i2w[:100])

    ## Set up the models

    img_enc = models.ImEncoder(in_size=(arg.img_size, arg.img_size))
    img_dec = models.ImDecoder(in_size=(arg.img_size, arg.img_size))

    embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=arg.embedding_size)

    seq_enc = models.SeqEncoder(vocab_size=vocab_size, embedding=embedding)
    seq_dec = models.SeqDecoder(vocab_size=vocab_size, embedding=embedding)

    if torch.cuda.is_available():
        img_enc.cuda()
        img_dec.cuda()
        seq_enc.cuda()
        seq_dec.cuda()

    #- The standard dataloader approach doesn't seem to work with the captions, so we'll do our own batching.
    #  It's a little slower, probably, but it won't be the bottleneck
    params = list(img_enc.parameters()) + list(img_dec.parameters())
    optimizer = Adam(params, lr=options.lr)

    instances_seen = 0

    for e in range(arg.epochs):
        print('epoch', e)
        for fr in tqdm.trange(0, len(coco_data), arg.batch_size):

            to = min(len(coco_data), fr + arg.batch_size)

            images = []
            captions = []

            for i in range(fr, to):
                images.append(coco_data[i][0].unsqueeze(0))
                captions.append(random.choice(coco_data[i][1])) # we choose one of the available captions at random

            imbatch = torch.cat(images, dim = 0)
            b, c, w, h = imbatch.size()

            capbatch = [] # to integer sequence
            for caption in captions:
                capbatch.append(util.intseq(caption, w2i))

            capbatch, lengths = util.pad(capbatch)

            # Created shifted versions
            b, s = capbatch.size()

            # Input for the decoder
            cap_teacher = torch.cat([torch.ones(b, 1, dtype=torch.long), capbatch], dim=1)
            cap_out     = torch.cat([capbatch, torch.zeros(b, 1, dtype=torch.long)], dim=1)

            lengths = torch.LongTensor(lengths)

            if torch.cuda.is_available():
                imbatch = imbatch.cuda()
                capbatch = capbatch.cuda()
                cap_teacher = cap_teacher.cuda()
                cap_out = cap_out.cuda()
                lengths = lengths.cuda()

            imbatch = Variable(imbatch)
            capbatch = Variable(capbatch)
            cap_teacher = Variable(cap_teacher)
            cap_out = Variable(cap_out)
            lengths = Variable(lengths)

            zimg = img_enc(imbatch)
            zcap = seq_enc(capbatch, lengths)

            kl_img = util.kl_loss(*zimg)
            kl_cap = util.kl_loss(*zcap)

            rec_img = img_dec(util.sample(*zimg))
            rec_img = binary_cross_entropy(rec_img, imbatch, reduce=False).view(b, -1).sum(dim=1)

            rec_cap = seq_dec(util.sample(*zcap), cap_teacher, lengths + 1)
            rec_cap = rec_cap.transpose(1, 2)

            rec_cap = nll_loss(rec_cap, cap_out, reduce=False).view(b, -1).sum(dim=1)

            loss_img = (rec_img + kl_img).mean()
            loss_cap = (rec_cap + kl_cap).mean()

            #- backward pass
            loss = loss_img + loss_cap

            loss.backward()
            optimizer.step()

            instances_seen += b

            tbw.add_scalar('score/img/kl', float(kl_img.mean()), instances_seen)
            tbw.add_scalar('score/img/rec', float(rec_img.mean()), instances_seen)
            tbw.add_scalar('score/cap/kl', float(kl_cap.mean()), instances_seen)
            tbw.add_scalar('score/cap/rec', float(rec_cap.mean()), instances_seen)
            tbw.add_scalar('score/loss', float(loss), instances_seen)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the embeddings.",
                        default=300, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate.",
                        default=0.0001, type=float)

    parser.add_argument("-I", "--image-size",
                        dest="img_size",
                        help="Size in pixels of on eof the sides of the (square) images.",
                        default=128, type=int)

    parser.add_argument("-w", "--max-vocab",
                        dest="max_vocab",
                        help="Maximum vocabulary.",
                        default=25000, type=int)

    parser.add_argument("-D", "--data-directory",
                        dest="data_dir",
                        help="Data directory",
                        default='./data', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/score', type=str)

    parser.add_argument("-C", "--cache-directory",
                        dest="cache_dir",
                        help="Dir for cache files (delete the dir to reconstruct)",
                        default='./cache', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)