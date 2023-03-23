import torch
import torch.nn as nn
import pdb
from utils import idx2onehot
import torch.nn.functional as F
class Classifier(nn.Module):
    def __init__(self,input_dim, num_labels=65):
        super().__init__()
        self.fc = nn.Linear(input_dim,num_labels)
        self.fc1 = nn.Linear(512,num_labels)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()
        self.relu = nn.ReLU()
    def forward(self, x):
        #o = self.logic(self.fc1(self.relu(self.fc(x))))
        o = self.logic(self.fc(x))
        return o
        
class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0, num_domains=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        _encoder_layer_sizes=encoder_layer_sizes[:] # list will be changed if it is changed in a function
        self.encoderS = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels,num_domains)
        self.encoderT = Encoder(
            _encoder_layer_sizes, latent_size, conditional, num_labels,num_domains)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels,num_domains)

    def forward(self, x_S, x_T, c_S=None, c_T=None):

        batch_size = x_S.size(0)

        means_S, log_var_S = self.encoderS(x_S, c_S)
        means_T, log_var_T = self.encoderT(x_T, c_T)

        std_S = torch.exp(0.5 * log_var_S)
        eps_S = torch.randn([batch_size, self.latent_size])
        z_S = eps_S * std_S + means_S
        recon_x_S = self.decoder(z_S, c_S)
        
        std_T = torch.exp(0.5 * log_var_T)
        eps_T = torch.randn([batch_size, self.latent_size])
        z_T = eps_T * std_T + means_T
        recon_x_T = self.decoder(z_T, c_T)

        return recon_x_S,recon_x_T, means_S,means_T, log_var_S,log_var_T, z_S,z_T

    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c)

        return recon_x

class VAE2(nn.Module):
    # One encoder one decoder, domain type as the last element of label indicator vector
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0,num_domains=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels, num_domains)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels, num_domains)
        self.conditional = conditional
    def forward(self, x, c=None,d=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c, d)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to('cuda')
        z = eps * std + means
        recon_x = self.decoder(z, c, d)
        recon_x2 = self.decoder(z, c, 1-d)
        
        if self.conditional:
            return recon_x,means,log_var,z
        return recon_x,recon_x2,means, log_var, z

    def inference(self, n=1, c=None, d=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c, d)

        return recon_x
        
class VAE2_1(nn.Module):
    # One encoder one decoder, domain vector as input of decoder
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, num_labels=0,num_domains=0,device='cuda'):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.device = device
        self.encoder = Encoder2_1(
            encoder_layer_sizes, latent_size, conditional, num_labels, num_domains)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels, num_domains, device=self.device)
        self.conditional = conditional
    def forward(self, x, c=None,d=None):

        batch_size = x.size(0)

        means, log_var = self.encoder(x, c, d)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = eps * std + means
        recon_x = self.decoder(z, c, d)
        recon_x2 = self.decoder(z, c, 1-d)
        
        if self.conditional:
            return recon_x,means,log_var,z
        return recon_x,recon_x2,means, log_var, z

    def inference(self, n=1, c=None, d=None):

        batch_size = n
        z = torch.randn([batch_size, self.latent_size])

        recon_x = self.decoder(z, c, d)

        return recon_x
        
class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, num_domains):

        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels
        self.num_domains = num_domains
        if self.conditional:
            layer_sizes[0] += num_labels
        if num_domains > 0:
            layer_sizes[0] += num_domains
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None, d=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        if self.num_domains>0:
            d = idx2onehot(d.cpu(), n=self.num_domains).to('cuda')
            x = torch.cat((x, d), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Encoder2_1(nn.Module):
    # For VAE2_1: domain information is not fed into the encoder
    def __init__(self, layer_sizes, latent_size, conditional, num_labels, num_domains):

        super().__init__()

        self.conditional = conditional
        self.num_labels = num_labels
        self.num_domains = num_domains
        if self.conditional:
            layer_sizes[0] += num_labels
        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None, d=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            x = torch.cat((x, c), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, num_domains, device='cuda'):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        self.num_labels = num_labels
        self.num_domains = num_domains
        self.device = device
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size
        if self.num_domains > 0:
            input_size += num_domains
            
        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c=None, d=None):

        if self.conditional:
            c = idx2onehot(c, n=self.num_labels)
            z = torch.cat((z, c), dim=-1)
        if self.num_domains > 0:
            d = idx2onehot(d.cpu(), n=self.num_domains).to(self.device)
            z = torch.cat((z,d), dim=-1)
        x = self.MLP(z)
        x = F.normalize(x, p=2, dim=1)
        return x
