import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

NUM_CLASSES = 1001

class RNNModel(nn.Module):
    def __init__(self, nlayers=3):
        super(RNNModel, self).__init__()
        hdim_rgb = 1024
        hdim_audio = 256

        out_hdim = nlayers*2*(hdim_rgb+hdim_audio)
        
        self.gru_rgb = nn.GRU(input_size=1024, hidden_size=hdim_rgb, num_layers=nlayers, bidirectional=True, batch_first=True)
        self.gru_audio = nn.GRU(input_size=128, hidden_size=hdim_audio, num_layers=nlayers, bidirectional=True, batch_first=True)
    
        self.fc = nn.Linear(out_hdim, NUM_CLASSES)
        self.gate_fc = nn.Linear(out_hdim, out_hdim)
        self.dropout = nn.Dropout(p=0.4)
        self.name = 'RNNModel_'+str(nlayers)

    def forward(self, rgb, audio):
        # out: (seq_len, batch, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        _, hrgb = self.gru_rgb(rgb)
        #print('hrgb:', hrgb.size())
        #print('_:', type(_), _.size())
        _, haudio = self.gru_audio(audio)
    
        h = torch.transpose(torch.cat((hrgb, haudio), 2), 0, 1).contiguous().view(rgb.size()[0], -1)
        #h = self.dropout(h)
        #h = F.dropout(h, p=0.6, training=self.training)
        #print('h:', h.size())
        gate = torch.sigmoid(self.gate_fc(h))
        h = gate * h

        h = F.dropout(h, p=0.6, training=self.training)
        

        return self.fc(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, nlayers=3, hdim_rgb=1024, hdim_audio=128, nhead=8, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.name = 'Transformer_gate_'+str(nlayers)
        self.dropout = dropout
        self.src_mask = None
        self.pos_encoder_rgb = PositionalEncoding(hdim_rgb, dropout)
        self.pos_encoder_audio = PositionalEncoding(hdim_audio, dropout)
        
        encoder_layers_rgb = TransformerEncoderLayer(hdim_rgb, nhead, hdim_rgb, dropout)
        encoder_layers_audio = TransformerEncoderLayer(hdim_audio, nhead, hdim_audio, dropout)

        self.transformer_rgb = TransformerEncoder(encoder_layers_rgb, nlayers)
        self.transformer_audio = TransformerEncoder(encoder_layers_audio, nlayers)

        self.fc = nn.Linear((hdim_audio+hdim_rgb)*5, NUM_CLASSES)

        self.gate_fc = nn.Linear((hdim_audio+hdim_rgb)*5, (hdim_audio+hdim_rgb)*5)
        

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, rgb, audio):
        rgb = rgb.transpose(0, 1)
        audio = audio.transpose(0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(rgb):
            device = rgb.device
            mask = self._generate_square_subsequent_mask(len(rgb)).to(device)
            self.src_mask = mask
            #print('mask:',  mask.size())
        rgb = self.pos_encoder_rgb(rgb)
        audio = self.pos_encoder_audio(audio)

        hrgb = self.transformer_rgb(rgb, self.src_mask)
        haudio = self.transformer_audio(audio, self.src_mask)

        #print(hrgb.size())
        h = torch.transpose(torch.cat((hrgb, haudio), 2), 0, 1).contiguous().view(rgb.size()[1], -1)

        gate = torch.sigmoid(self.gate_fc(h))
        h = gate * h

        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc(h)


def create_model(args):
    #model = RNNModel(args.nlayers)
    model = TransformerModel(args.nlayers)
    model_file = os.path.join(settings.MODEL_DIR, model.name, args.ckp_name)

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    print('model file: {}, exist: {}'.format(model_file, os.path.exists(model_file)))

    if args.predict and (not os.path.exists(model_file)):
        raise AttributeError('model file does not exist: {}'.format(model_file))

    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))

    return model, model_file

def test_forward():
    model = RNNModel().cuda()
    rgb = torch.randn(4,300,1024).cuda()
    audio = torch.randn(4,300,128).cuda()
    out = model(rgb, audio).squeeze()
    print(out.size())

def test_transformer():
    model = TransformerModel().cuda()
    #x = torch.tensor([[1,2,3,4,5]]*100).cuda().transpose(0,1)
    rgb = torch.randn(4, 5, 1024).cuda()
    audio = torch.randn(4, 5, 128).cuda()
    out = model(rgb, audio)
    print(out.size())

if __name__ == '__main__':
    #test_forward()
    test_transformer()
