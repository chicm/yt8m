import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings

NUM_CLASSES = 1000

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
        self.name = 'RNNModel_'+str(nlayers)

    def forward(self, rgb, audio):
        # out: (seq_len, batch, hidden_size * num_directions)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        _, hrgb = self.gru_rgb(rgb)
        #print('hrgb:', hrgb.size())
        #print('_:', type(_), _.size())
        _, haudio = self.gru_audio(audio)
    
        h = torch.transpose(torch.cat((hrgb, haudio), 2), 0, 1).contiguous().view(rgb.size()[0], -1)
        gate = torch.sigmoid(self.gate_fc(h))
        h = gate * h

        h = F.dropout(h, p=0.4, training=self.training)
        
        return self.fc(h)

def create_model(args):
    model = RNNModel(args.nlayers)
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

if __name__ == '__main__':
    test_forward()
