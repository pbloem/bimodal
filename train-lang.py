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

REP = 3
TEMPS = [0.0, 0.1, 1.0]

def go(arg):

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    with open(arg.data, 'r') as file:
        lines = file.readlines()

    print('Creating word indices')

    dist = Counter()

    for line in lines:
        dist.update(util.tokenize(line))

    vocab = dist.most_common(arg.max_vocab - len(EXTRA_SYMBOLS))

    i2w = EXTRA_SYMBOLS + [w[0] for w in vocab]
    w2i = {word:ix for ix, word in enumerate(i2w)}

    data = []
    suff = [EOS] if arg.add_eos else []

    for line in lines:
        words = util.tokenize(line)
        indices = []
        for word in words:
            if word in w2i:
                indices.append(w2i[word])
            else:
                indices.add(UNK)

        if len(indices) > 0:
            data.append(indices + suff)

    data.sort(key= lambda x : len(x))

    vocab_size = len(i2w)
    print('vocabulary size', vocab_size)
    print('top 100 words:', i2w[:100])
    print('sentence lengths ', [len(s) for s in data])

    def decode(indices):

        sentence = ''
        for id in indices:
                if id == PAD:
                    break
                sentence += i2w[id] + ' '

        return sentence

    s = random.choice(data)
    print('random sentence', s)
    print('               ', decode(s))

    ## Set up the model

    embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=arg.embedding_size)

    seq_enc = models.SeqEncoder(vocab_size=vocab_size, embedding=embedding, zsize=arg.latent_size)
    seq_dec = models.SeqDecoder(vocab_size=vocab_size, embedding=embedding, zsize=arg.latent_size)

    mods = [seq_enc, seq_dec]

    if torch.cuda.is_available():
        for model in mods:
            model.cuda()

    params = []
    for model in mods:
        params.extend(model.parameters())
    optimizer = Adam(params, lr=arg.lr)

    instances_seen = 0

    for e in range(arg.epochs):
        if arg.annealing_mode == None or arg.annealing_mode == 'none':
            weight = 1
        elif arg.annealing_mode == 'linear':
            weight = util.lin_anneal(e, arg.epochs)
        elif arg.annealing_mode == 'logistic':
            weight = util.log_anneal(e, arg.epochs)
        else:
            raise Exception('Annea;ing mode {} not recognized'.format(arg.annealing_mode))

        print('Epoch {}, setting KL weight to {}'.format(e, weight))
        for fr in tqdm.trange(0, len(data), arg.batch_size):
            if arg.instance_limit is not None and fr > arg.instance_limit:
                break

            to = min(len(data), fr + arg.batch_size)

            batch = data[fr:to]
            batch, lengths = util.pad(batch)

            # Created shifted versions
            b, s = batch.size()

            # Input for the decoder
            batch_teacher = torch.cat([torch.ones(b, 1, dtype=torch.long), batch], dim=1)
            batch_out     = torch.cat([batch, torch.zeros(b, 1, dtype=torch.long)], dim=1)

            lengths = torch.LongTensor(lengths)

            if torch.cuda.is_available():

                batch = batch.cuda()
                batch_teacher = batch_teacher.cuda()
                batch_out = batch_out.cuda()

                lengths = lengths.cuda()

            batch = Variable(batch)
            batch_teacher = Variable(batch_teacher)
            batch_out = Variable(batch_out)
            lengths = Variable(lengths)

            z = seq_enc(batch, lengths)

            kl = util.kl_loss(*z)

            rec = seq_dec(util.sample(*z), batch_teacher, lengths + 1)
            rec = rec.transpose(1, 2)
            rl  = nll_loss(rec, batch_out, reduce=False).view(b, -1)

            rl = rl.sum(dim=1)

            loss = (rl + weight * kl).mean()

            #- backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            instances_seen += b

            tbw.add_scalar('lang/kl', float(kl.mean()), instances_seen)
            tbw.add_scalar('lang/rec', float(rl.mean()), instances_seen)
            tbw.add_scalar('lang', float(loss), instances_seen)

        # Interpolate
        for r in range(REP):

            print('Interpolation, repeat', r)

            z1, z2 = torch.randn(2, arg.latent_size)
            if torch.cuda.is_available():
                z1, z2 = z1.cuda(), z2.cuda()

            zs = util.slerp(z1, z2, 10)

            print('== sentences (temp={}) =='.format(TEMPS[r]))
            # sentences = seq_dec.sample(z=zs, temperature=TEMPS[r])
            sentences = seq_dec.sample_old(z=zs)

            for s in sentences:
                print('   ', decode(s))

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-a", "--annealing-mode",
                        dest="annealing_mode",
                        help="Annealing mode: none, logistic, linear.",
                        default=None, type=str)

    parser.add_argument("-A","--add-eos",
                        dest="add_eos",
                        help="Add eos token to end of sentence",
                        action="store_true")

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=150, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="Size of the batches.",
                        default=32, type=int)

    parser.add_argument("-L", "--latent-size",
                        dest="latent_size",
                        help="Size of the latent representations.",
                        default=32, type=int)

    parser.add_argument("-E", "--embedding-size",
                        dest="embedding_size",
                        help="Size of the embeddings.",
                        default=300, type=int)

    parser.add_argument("--limit",
                        dest="instance_limit",
                        help="Limit on the number of instances seen per batch (for debugging).",
                        default=None, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate.",
                        default=0.001, type=float)

    parser.add_argument("-w", "--max-vocab",
                        dest="max_vocab",
                        help="Maximum vocabulary.",
                        default=10000, type=int)

    parser.add_argument("-D", "--data-file",
                        dest="data",
                        help="Data file",
                        default='./data/', type=str)

    parser.add_argument("-T", "--tb-directory",
                        dest="tb_dir",
                        help="Tensorboard directory",
                        default='./runs/train-lang', type=str)

    parser.add_argument("-C", "--cache-directory",
                        dest="cache_dir",
                        help="Dir for cache files (delete the dir to reconstruct)",
                        default='./cache', type=str)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(options)