import torch
import torch.nn


import torch
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy, relu, log_softmax
from torch.nn import \
    Module, Conv2d, ConvTranspose2d, MaxPool2d, Linear, Sequential, ReLU, Sigmoid, Upsample, \
    Embedding, Dropout, GRU
import torch.nn.utils.rnn as rnn_utils

from torch.autograd import Variable

import util
from util import to_var
from util import PAD, SOS, EOS, UNK, EXTRA_SYMBOLS

"""
Code for the Sequence VAE borrowed from https://github.com/timbmg/Sentence-VAE

"""

class ImEncoder(Module):

    def __init__(self, in_size, zsize=32, use_res=False, use_bn=False, depth=0):
        super().__init__()
        self.zsize = zsize

        # - channel sizes
        a, b, c = 8, 32, 128

        # - Encoder
        modules = [
            util.Block(3, a, use_res=use_res, batch_norm=use_bn),
            MaxPool2d((4, 4)),
            util.Block(a, b, use_res=use_res, batch_norm=use_bn),
            MaxPool2d((4, 4)),
            util.Block(b, c, use_res=use_res, batch_norm=use_bn),
            MaxPool2d((4, 4)),
        ]

        for i in range(depth):
            modules.append( util.Block(c, c, use_res=use_res, batch_norm=use_bn))

        modules.extend([
            util.Flatten(),
            Linear((in_size[0] // 64) * (in_size[1] // 64) * c, zsize * 2)
        ])

        self.encoder = Sequential(*modules)

    def forward(self, image):

        zcomb = self.encoder(image)
        return zcomb[:, :self.zsize], zcomb[:, self.zsize:]

class ImDecoder(Module):

    def __init__(self, in_size, zsize=32, use_res=False, use_bn=False, depth=0):
        super().__init__()

        self.zsize = zsize

        # - channel sizes
        a, b, c = 8, 32, 128

        #- Decoder
        upmode = 'bilinear'
        modules = [
            Linear(zsize, (in_size[0] // 64) * (in_size[1] // 64) * c), ReLU(),
            util.Reshape((c, in_size[0] // 64, in_size[1] // 64))
        ]

        for _ in range(depth):
            modules.append( util.Block(c, c, deconv=True, use_res=use_res, batch_norm=use_bn) )


        modules.extend([
            Upsample(scale_factor=4, mode=upmode),
            util.Block(c, c, deconv=True, use_res=use_res, batch_norm=use_bn),
            Upsample(scale_factor=4, mode=upmode),
            util.Block(c, b, deconv=True, use_res=use_res, batch_norm=use_bn),
            Upsample(scale_factor=4, mode=upmode),
            util.Block(b, a, deconv=True, use_res=use_res, batch_norm=use_bn),
            ConvTranspose2d(a, 3, kernel_size=1, padding=0),
            Sigmoid()
        ])

        self.decoder = Sequential(*modules)

    def forward(self, zsample):

        return self.decoder(zsample)

class SeqEncoder(Module):

    def __init__(self,  vocab_size, zsize=32, embedding_size=300,hidden_size=256, embedding=None):
        super().__init__()

        self.zsize = zsize
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = Embedding(vocab_size, embedding_size) if embedding is None else embedding

        self.encoder_rnn = GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.hidden_factor = 2

        self.fromhidden = Linear(hidden_size * 2, 2 * zsize)

    def forward(self, sequence, lengths):

        batch_size = sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        input_sequence = sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)

        zcomb = self.fromhidden(hidden)
        return zcomb[:, :self.zsize], zcomb[:, self.zsize:]

class SeqDecoder(Module):

    def __init__(self, vocab_size, zsize=32, dropout=0.5, hidden_size=256, embedding_size=300, embedding=None):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.zsize = zsize
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # I think this is the wrong kind of dropout... (maybe this actually works better?)
        self.word_dropout = Dropout(p=dropout)

        self.embedding = Embedding(vocab_size, embedding_size) if embedding is None else embedding

        self.decoder_rnn = GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.outputs2vocab = Linear(hidden_size * 2, vocab_size)

        self.tohidden = Linear(zsize, hidden_size * 2)

    def forward(self, zsample, sequence, lengths):

        """
        Outputs log-softmax

        :param zsample:
        :param sequence:
        :param lengths:
        :return:
        """

        b, _ = zsample.size()

        sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
        input_sequence = sequence[sorted_idx]

        hidden = self.tohidden(zsample)

        hidden = hidden.view(2, b, self.hidden_size)

        # decoder input
        input_embedding = self.embedding(input_sequence)
        input_embedding = self.word_dropout(input_embedding)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0].contiguous()

        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, h = padded_outputs.size()

        # project outputs to vocab
        voc  = self.outputs2vocab(padded_outputs.view(-1, h)).view(b, s, self.vocab_size)
        logp = log_softmax(voc, dim=-1)

        return logp

    def sample(self, n=4, z=None, max_length=60):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.tohidden(z)

        hidden = hidden.view(2, batch_size, self.hidden_size)

        # required for dynamic stopping of sentence generation
        sequence_idx     = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask    = torch.ones(batch_size, out=self.tensor()).byte()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, max_length).fill_(PAD).long()

        t=0
        while t < max_length and len(running_seqs) > 0:

            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(SOS).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update global running sequence
            sequence_mask[sequence_running] = (input_sequence != EOS).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != EOS).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs.size()) > 0:

                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
