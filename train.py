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
from enum import Enum

from tensorboardX import SummaryWriter

REP = 3
TEMPS = [0.0, 0.1, 1.0]

class Mode(Enum):
    independent = 'independent'
    coupled = 'coupled'
    style = 'style'

    def __str__(self):
        return self.value

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

    def decode(indices):

        sentence = ''
        for id in indices:
                # if id == PAD:
                #     break
                sentence += i2w[id] + ' '

        return sentence

    ## Set up the models
    embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=arg.embedding_size)

    if arg.mode != Mode.style:
        img_enc = models.ImEncoder(in_size=(arg.img_size, arg.img_size), zsize=arg.latent_size)
        img_dec = models.ImDecoder(in_size=(arg.img_size, arg.img_size), zsize=arg.latent_size)

        seq_enc = models.SeqEncoder(vocab_size=vocab_size, embedding=embedding, zsize=arg.latent_size)
        seq_dec = models.SeqDecoder(vocab_size=vocab_size, embedding=embedding, zsize=arg.latent_size)

        mods = [img_enc, img_dec, seq_enc, seq_dec]
    else:
        img_enc = models.ImEncoder(in_size=(arg.img_size, arg.img_size), zsize=arg.latent_size)
        img_sty = models.ImEncoder(in_size=(arg.img_size, arg.img_size), zsize=arg.latent_size)
        img_dec = models.ImDecoder(in_size=(arg.img_size, arg.img_size), zsize=arg.latent_size * 2)

        seq_enc = models.SeqEncoder(vocab_size=vocab_size, embedding=embedding, zsize=arg.latent_size)
        seq_sty = models.SeqEncoder(vocab_size=vocab_size, embedding=embedding, zsize=arg.latent_size)
        seq_dec = models.SeqDecoder(vocab_size=vocab_size, embedding=embedding, zsize=arg.latent_size * 2)

        mods = [img_enc, img_dec, img_sty, seq_enc, seq_dec, seq_sty]

    if torch.cuda.is_available():
        for model in mods:
            model.cuda()

    #- The standard dataloader approach doesn't seem to work with the captions, so we'll do our own batching.
    #  It's a little slower, probably, but it won't be the bottleneck
    params = []
    for model in mods:
        params.extend(model.parameters())
    optimizer = Adam(params, lr=arg.lr)

    instances_seen = 0

    for e in range(arg.epochs):
        print('epoch', e)
        for fr in tqdm.trange(0, len(coco_data), arg.batch_size):
            if arg.instance_limit is not None and fr > arg.instance_limit:
                break

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
                capbatch.append(util.intseq(util.tokenize(caption), w2i))

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

            zimg_sample = util.sample(*zimg)
            zcap_sample = util.sample(*zcap)

            if arg.mode == Mode.style:
                zimg_sty = img_sty(imbatch)
                zcap_sty = seq_sty(capbatch, lengths)

                kl_img_sty = util.kl_loss(*zimg_sty)
                kl_cap_sty = util.kl_loss(*zcap_sty)

                zimg_sample_sty = util.sample(*zimg_sty)
                zcap_sample_sty = util.sample(*zcap_sty)

                zimg_sample = torch.cat([zimg_sample, zimg_sample_sty], dim=1)
                zcap_sample = torch.cat([zcap_sample, zcap_sample_sty], dim=1)

            rec_imgimg = img_dec(zimg_sample)
            rl_imgimg = binary_cross_entropy(rec_imgimg, imbatch, reduce=False).view(b, -1).sum(dim=1)

            rec_capcap = seq_dec(zcap_sample, cap_teacher, lengths + 1).transpose(1, 2)
            rl_capcap = nll_loss(rec_capcap, cap_out, reduce=False).view(b, -1).sum(dim=1)

            if arg.mode != Mode.independent:
                rec_capimg = img_dec(zcap_sample)
                rl_capimg = binary_cross_entropy(rec_capimg, imbatch, reduce=False).view(b, -1).sum(dim=1)

                rec_imgcap = seq_dec(zimg_sample, cap_teacher, lengths + 1).transpose(1, 2)
                rl_imgcap = nll_loss(rec_imgcap, cap_out, reduce=False).view(b, -1).sum(dim=1)

            loss_img = rl_imgimg + kl_img
            loss_cap = rl_capcap + kl_cap

            if arg.mode == Mode.coupled:
                loss_img = loss_img + rl_capimg + kl_img
                loss_cap = loss_cap + rl_imgcap + kl_cap

            if arg.mode == Mode.style:
                loss_img = loss_img + kl_img_sty
                loss_cap = loss_cap + kl_cap_sty

            loss = loss_img.mean() + loss_cap.mean()

            #- backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            instances_seen += b

            tbw.add_scalar('score/img/kl', float(kl_img.mean()), instances_seen)
            tbw.add_scalar('score/imgimg/rec', float(rl_imgimg.mean()), instances_seen)
            tbw.add_scalar('score/cap/kl', float(kl_cap.mean()), instances_seen)
            tbw.add_scalar('score/capcap/rec', float(rl_capcap.mean()), instances_seen)
            tbw.add_scalar('score/loss', float(loss), instances_seen)

            if arg.mode != Mode.independent:
                tbw.add_scalar('score/capimg/rec', float(rl_capimg.mean()), instances_seen)
                tbw.add_scalar('score/imgcap/rec', float(rl_imgcap.mean()), instances_seen)

        # Interpolate
        zpairs = []
        for r in range(REP):

            print('Interpolation, repeat', r)

            l = arg.latent_size if arg.mode != Mode.style else arg.latent_size * 2
            z1, z2 = torch.randn(2, l)
            if torch.cuda.is_available():
                z1, z2 = z1.cuda(), z2.cuda()

            zpairs.append((z1, z2))

            zs = util.slerp(z1, z2, 10)

            print('== sentences (temp={}) =='.format(TEMPS[r]))
            sentences = seq_dec.sample(z=zs, temperature=TEMPS[r])

            for s in sentences:
                print('   ', decode(s))

        print('== images ==')

        util.interpolate(zpairs, img_dec, name='interpolate.{}'.format(e))

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-m", "--mode",
                        dest="mode",
                        help="Mode. independent: trains fully separate autoencoders for images and language. shared: couples the latent space of the two autoencoders. style: uses separate encoders to capture style.",
                        default=Mode.independent, type=Mode, choices=list(Mode))

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