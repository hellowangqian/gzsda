import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy,scipy.io
from sklearn.preprocessing import normalize
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from collections import defaultdict
import numpy as np
from models import VAE2,Classifier

domainSet =['A','D','W']
class TwoModalDataset(Dataset):
    def __init__(self,phase='train',sourceDomainIndex=0, targetDomainIndex = 0,trialIndex=0):
        self.load_mat(sourceDomainIndex,targetDomainIndex,trialIndex)
        self.phase = phase
        if self.phase == 'train':
            flag = 1
        #if self.phase == 'val':
        #    flag = 2
        if self.phase == 'test':
            flag = 2
        #self.feature_A = self.feature_A
        self.feature_B = self.feature_B[self.splitFlag_B==flag,]
        #self.label_A = self.label_A
        self.label_B = self.label_B[self.splitFlag_B==flag]
        if self.phase == 'train':
            #self.features = self.feature_B
            #self.labels = self.label_B
            self.features = np.concatenate((self.feature_A,self.feature_B))
            self.labels = np.concatenate((self.label_A,self.label_B))
        if self.phase == 'val' or self.phase == 'test':
            self.features = self.feature_B
            self.labels = self.label_B
            
    def load_mat(self,sourceDomainIndex=0, targetDomainIndex=0,trialIndex=0):
        # load features and labels
        data_dir = './data/Office31/'
        data_A = scipy.io.loadmat(data_dir+'office-'+domainSet[sourceDomainIndex]+'-resnet50-noft.mat')
        feature_A = data_A['resnet50_features'][:,:,0,0]
        self.feature_A = normalize(feature_A,norm='l2')
        self.label_A = data_A['labels'][0,]
        self.num_class = len(np.unique(self.label_A))
        data_B = scipy.io.loadmat(data_dir+'office-'+domainSet[targetDomainIndex]+'-resnet50-noft.mat')
        feature_B = data_B['resnet50_features'][:,:,0,0]
        self.feature_B = normalize(feature_B,norm='l2')
        self.label_B = data_B['labels'][0,]
        dataSplit = scipy.io.loadmat(data_dir+'instanceSplit_office31_unseen15_20221027.mat')# default data split for office-home, 30 unseen classes
        self.splitFlag_B = dataSplit['targetDomain_splitFlag'][0,trialIndex][0,targetDomainIndex][0,] # [0, index of trial][0, index of domain], 1--train, 2--test, 0--not used
        self.unseenClass_B = dataSplit['targetDomain_unseenClass'][0,trialIndex][0,targetDomainIndex][0,]
    def __len__(self):
        if self.phase == 'train': #or self.phase == 'val':
            return self.feature_A.shape[0]
        if self.phase == 'test':
            return self.feature_B.shape[0]
    def __getitem__(self,idx):

        # return a pair of regular and xray image features, which are paired randomly
        if self.phase == 'test':
            idx_B = idx
            return self.feature_B[idx_B,:],self.label_B[idx_B]
        label = self.label_A[idx]
        indicesB_this_label = np.argwhere(self.label_B == label)
        if len(indicesB_this_label) > 0:
            idx_B = np.random.choice(indicesB_this_label[:,0])
            return self.feature_A[idx,:], self.feature_B[idx_B,:],self.label_A[idx],self.label_B[idx_B]
        else:
            return self.feature_A[idx,:], np.zeros_like(self.feature_A[idx,:]), self.label_A[idx], np.ones_like(self.label_A[idx]) * -1


def test_model(model,dataset,dataloader,device,model_type='knn'):
    since = time.time()
    
    num_class = dataset.num_class
    running_corrects = np.zeros((num_class,))
    num_sample_per_class = np.zeros((num_class,))
    # Iterate over data.
    for index, (features,labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            if model_type=='knn':
                preds = model.predict(features)
            if model_type=='mlp':
                model.eval()
                preds = model(features)
                preds = preds.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
            if index == 0:
                outputs_test = preds
                labels_test = labels
            else:
                outputs_test = np.concatenate((outputs_test, preds), 0)
                labels_test = np.concatenate((labels_test, labels), 0)
        if model_type=='mlp':
            preds = np.argmax(outputs_test,1)
        if model_type=='knn':
            preds = outputs_test
   
    for i in range(len(labels_test)):
        num_sample_per_class[labels_test[i]] += 1
        if preds[i]==labels_test[i]:
            running_corrects[labels_test[i]] += 1

    acc_per_class = running_corrects / num_sample_per_class
    acc = np.mean(acc_per_class)
    acc_seen = np.mean(acc_per_class[dataset.unseenClass_B==0])
    acc_unseen = np.mean(acc_per_class[dataset.unseenClass_B==1])
    time_elapsed = time.time() - since
    #print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('per-class acc:{:2.4f}, seen acc:{:2.4f}, unseen acc:{:2.4f}'.format(acc,acc_seen,acc_unseen))
    return acc_per_class,acc,acc_seen,acc_unseen

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    datasets = {x: TwoModalDataset(phase=x,sourceDomainIndex=args.sourceDomainIndex, targetDomainIndex=args.targetDomainIndex,trialIndex=args.trialIndex) for x in ['train','test']}
    data_loaders={}
    data_loaders['train'] = DataLoader(dataset=datasets['train'], batch_size=args.batch_size, shuffle=True)
    data_loaders['trainall'] = DataLoader(dataset=datasets['train'], batch_size=len(datasets['train']), shuffle=True)
    data_loaders['test'] = DataLoader(dataset=datasets['test'], batch_size=len(datasets['test']), shuffle=False)

    def loss_fn(recon_xS, xS, recon_xT, xT, meanS, log_varS, meanT, log_varT):
        criterion = torch.nn.MSELoss(size_average=False)
        reconstruction_loss = criterion(recon_xS, xS) + criterion(recon_xT, xT)
        KLD = -0.5 * torch.sum(1 + log_varS - meanS.pow(2) - log_varS.exp())  -0.5 * torch.sum(1 + log_varT - meanT.pow(2) - log_varT.exp())
        #print(reconstruction_loss, KLD)
        return (reconstruction_loss + 1*KLD) / xS.size(0)

    def loss_fn2(recon_xS,recon_xS2, xS, recon_xT,recon_xT2, xT, meanS, log_varS, meanT, log_varT,yT,epoch):
        criterion = torch.nn.MSELoss(size_average=False)
        mask = yT!=-1
        reconstruction_loss = criterion(recon_xS, xS) + criterion(recon_xT[mask,:], xT[mask,:])
        cross_reconstruction_loss = criterion(recon_xS2[mask,:], xT[mask,:]) + criterion(recon_xT2[mask,:], xS[mask,:])
        KLD = -0.5 * torch.sum(1 + log_varS - meanS.pow(2) - log_varS.exp())  -0.5 * torch.sum(1 + log_varT[mask,:] - meanT[mask,:].pow(2) - log_varT[mask,:].exp())
        weight = epoch*5e-4
        return (reconstruction_loss + 1*cross_reconstruction_loss + weight*KLD) / xS.size(0)


    vae = VAE2(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_domains = 2,
        num_labels=31 if args.conditional else 0).to(device)    
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    ############################################################
    # train CVAE
    ############################################################
    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):

            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)

            if args.conditional:
                recon_xS, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
                recon_xT, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
                loss = loss_fn(recon_xS, xS, recon_xT, xT, meanS, log_varS, meanT, log_varT)
            else:
                recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
                recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
                #loss = loss_fn(recon_xS, xS, recon_xT, xT, meanS, log_varS, meanT, log_varT)
                loss = loss_fn2(recon_xS, recon_xS2, xS, recon_xT,recon_xT2, xT, meanS, log_varS, meanT, log_varT,yT,epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()


            #print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(epoch, args.epochs, iteration, len(data_loaders['train'])-1, loss.item()))
    ############################################################
    #Generating pseudo training samples and train/test a classifier
    ############################################################
    def next_batch(vae,batch_size=64):
        y = np.random.randint(31,size=batch_size)
        y = torch.from_numpy(y)        
        pseudo_x = vae.inference(n=batch_size,c=y,d = torch.ones_like(y))
        return pseudo_x, y
    def generate_z(xS,xT,vae):
        vae.eval()
        recon_xS, recon_xS2, meanS, log_varS, zS = vae(xS, yS, d=torch.zeros_like(yS).to(device))
        recon_xT, recon_xT2, meanT, log_varT, zT = vae(xT, yT, d=torch.ones_like(yT).to(device))
        return recon_xS2, recon_xT2

    classifier = Classifier(input_dim=2048,num_labels=31).to(device) # train and test a classifier
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)
    scheduler_cls = torch.optim.lr_scheduler.StepLR(optimizer_cls, step_size=25, gamma=0.1)
    num_epochs = 50
    acc_per_class = np.zeros((num_epochs,31))
    acc = np.zeros((num_epochs,))
    acc_seen = np.zeros((num_epochs,))
    acc_unseen = np.zeros((num_epochs,))
    for epoch in range(num_epochs):
        #print(epoch)
        for iteration, (xS,xT,yS,yT) in enumerate(data_loaders['train']):
            xS,xT,yS,yT = xS.to(device), xT.to(device), yS.to(device), yT.to(device)
            recon_xS,recon_xT = generate_z(xS,xT,vae)
            mask = yT!=-1
            xT = xT[mask,:]
            yT = yT[mask]
            recon_xT = recon_xT[mask,:]
            xtrain = torch.cat((xS,xT,recon_xS,recon_xT),dim=0)
            ytrain = torch.cat((yS,yT, yS, yT),dim=0)
            output = classifier(xtrain)
            loss_cls = classifier.lossfunction(output, ytrain)
            optimizer_cls.zero_grad()
            loss_cls.backward()
            optimizer_cls.step()
            #print("Epoch {:02d}/{:02d}, Loss {:9.4f}".format(epoch, args.epochs, loss_cls.item()))
            # test
        scheduler_cls.step()
        acc_per_class[epoch,],acc[epoch],acc_seen[epoch],acc_unseen[epoch] = test_model(classifier, datasets['test'], data_loaders['test'],device,model_type='mlp')
    scipy.io.savemat('./results20211012/'+args.filename+'.mat',mdict={'acc_per_class':acc_per_class,'acc':acc,'acc_seen':acc_seen,'acc_unseen':acc_unseen})


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[2048, 512])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[512, 2048])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--sourceDomainIndex", type=int, default=1)
    parser.add_argument("--targetDomainIndex", type=int, default=0)
    parser.add_argument("--trialIndex", type=int, default=0)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()
    
    source = domainSet[args.sourceDomainIndex]
    target = domainSet[args.targetDomainIndex]
    args.filename = 'office31-15-'+source+'-'+target+'-trial'+str(args.trialIndex)+'-vaeEpochs-'+str(args.epochs)+'-latSize-'+str(args.latent_size)+'lr'+str(args.learning_rate)
    print(args.filename)
    main(args)
