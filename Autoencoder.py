import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
import torch.utils.data as Data
import numpy as np
import copy
import torch.nn.utils.prune as prune
import time

class MyAE(nn.Module):
    def __init__(self):
        super(MyAE,self).__init__()
        self.fc1=nn.Linear(1024,328,bias=False)
        self.fl1=nn.ReLU()
        self.fc2=nn.Linear(328,75,bias=False)
        self.fl2=nn.ReLU()
        self.fc3=nn.Linear(75,328,bias=False)
        self.fl3=nn.ReLU()
        self.fc4=nn.Linear(328,1024,bias=False)
    def forward(self,x):
        x=self.fc1(x)
        x=self.fl1(x)
        x=self.fc2(x)
        x=self.fl2(x)
        x=self.fc3(x)
        x=self.fl3(x)
        x=self.fc4(x)
        return x
    def first(self,x):
        x=self.fc1(x)
        return x
    def mid(self,x):
        x=self.fc1(x)
        x=self.fl1(x)
        x=self.fc2(x)
        return x

def train(AE,train_loader,device,verbose=False):
    for param in AE.parameters():
        param.requires_grad=True
    optimizer=optim.Adam(AE.parameters(),lr=0.00005)
    E=1
    lam=0.05
    for epoch in range(E):
        for idx,(b_x,b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_z=b_x.reshape([-1,1024])
            optimizer.zero_grad()
            l=torch.sum((b_z-AE(b_z))**2)+lam*torch.sum((AE.mid(b_z))**2)
            l.backward()
            optimizer.step()
            if idx%100==0 and verbose:
                print("Epoch=%i,Index=%i,Loss=%f"%(epoch,idx,float(l.detach())/bs))
    return True

def ptrain(AE,train_loader,device,verbose=False):
    AE=myprune(AE,0.05,device)
    for param in AE.parameters():
        param.requires_grad=True
    optimizer=optim.Adam(AE.parameters(),lr=0.00001)
    E=1
    lam=0.05
    for epoch in range(E):
        for idx,(b_x,b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_z=b_x.reshape([-1,1024])
            optimizer.zero_grad()
            l=torch.sum((b_z-AE(b_z))**2)+lam*torch.sum((AE.mid(b_z))**2)
            l.backward()
            optimizer.step()
            if idx%100==0 and verbose:
                print("Epoch=%i,Index=%i,Loss=%f"%(epoch,idx,float(l.detach())/bs))
    return True

def test(AE,train_loader,device):
    l=0
    for idx,(b_x,b_y) in enumerate(train_loader):
        b_x=b_x.to(device)
        b_z=b_x.reshape([-1,1024])
        l=l+torch.sum((b_z-AE(b_z))**2)
    l=l/60000/1024
    return float(l.detach().cpu())
 
def myprune(model,rate,device):
    local_model=copy.deepcopy(model)
    local_model=local_model.to(device)
    module1=local_model.fc1
    prune.random_unstructured(module1,name="weight",amount=rate)
    module2=local_model.fc2
    prune.random_unstructured(module2,name="weight",amount=rate)
    prune.remove(module1,"weight")
    prune.remove(module2,"weight")
    return local_model

class UCVerifier(nn.Module):
    def __init__(self,r=0.1):
        super(UCVerifier,self).__init__()
        self.mask=torch.randint(0,1,size=(75,328))
        self.mask.requires_grad=False
        self.r=r
        for i in range(75):
            for j in range(328):
                if torch.rand(size=(1,1))[0][0]<=r:
                    self.mask[i][j]=1
        self.carrier=torch.randn(size=(75,328))
    def connect(self,AE):
        c=AE.state_dict()["fc2.weight"].detach().cpu()
        e=(c-self.carrier)*self.mask 
        e=torch.sum(e**2)
        e=e/(75*328*self.r)
        return e
    def fit(self,AE):
        c=AE.state_dict()["fc2.weight"].detach().cpu()
        self.carrier=copy.deepcopy(c)
        return True

def UCVerify(AE,V):
    return float(V.connect(AE))

class DSVerifier(nn.Module):
    def __init__(self,T):
        super(DSVerifier,self).__init__()
        self.T=T
        self.mask1=torch.zeros(size=(100,328))
        self.mask2=torch.zeros(size=(100,75))
        self.median1=0
        self.median2=0
    def connect(self,AE):
        AE=AE.to(torch.device("cpu"))
        m1=AE.first(T)
        m2=AE.mid(T)
        m1=m1.detach()
        m2=m2.detach()
        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                if m1[i][j]<=self.median1:
                    m1[i][j]=0
                else:
                    m1[i][j]=1
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                if m2[i][j]<=self.median2:
                    m2[i][j]=0
                else:
                    m2[i][j]=1
        a=torch.sum((m1-self.mask1)**2)+torch.sum((m2-self.mask2)**2)
        return a/(32800+7500)
    def fit(self,AE):
        AE=AE.to(torch.device("cpu"))
        m1=AE.first(T)
        m2=AE.mid(T)
        m1=m1.detach()
        m2=m2.detach()
        self.median1=torch.median(m1)
        self.median2=torch.median(m2)
        for i in range(m1.shape[0]):
            for j in range(m1.shape[1]):
                if m1[i][j]<=self.median1:
                    self.mask1[i][j]=0
                else:
                    self.mask1[i][j]=1
        for i in range(m2.shape[0]):
            for j in range(m2.shape[1]):
                if m2[i][j]<=self.median2:
                    self.mask2[i][j]=0
                else:
                    self.mask2[i][j]=1
        return True

def DSVerify(AE,V):
    return float(V.connect(AE))

class DJVerifier(nn.Module):
    def __init__(self,T):
        super(DJVerifier,self).__init__()
        self.T=T
        self.vmask1=torch.zeros(size=(100,328))
        self.vmask2=torch.zeros(size=(100,75))
        self.amask1=torch.zeros(size=(100,328))
        self.amask2=torch.zeros(size=(100,75))
    def connect(self,AE):
        AE=AE.to(torch.device("cpu"))
        f=AE.first(self.T)
        m=AE.mid(self.T)
        f=f.detach().cpu()
        m=m.detach().cpu()
        fmedian=torch.median(f)
        mmedian=torch.median(m)
        p=torch.norm(f-self.vmask1,2)+torch.norm(m-self.vmask2,2)
        p=p.detach().cpu()
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f[i][j]<=fmedian:
                    f[i][j]=0
                else:
                    f[i][j]=1
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i][j]<=mmedian:
                    m[i][j]=0
                else:
                    m[i][j]=1
        q=torch.norm(f-self.amask1,2)+torch.norm(m-self.amask2,2)
        p=p/(32800+7500)
        q=q/(32800+7500)
        return p,q
    def fit(self,AE):
        AE=AE.to(torch.device("cpu"))
        f=AE.first(self.T)
        f=f.detach()
        m=AE.mid(self.T)
        m=m.detach()
        self.vmask1=f.detach().cpu()
        self.amask1=copy.deepcopy(f)
        self.vmask2=m.detach().cpu()
        self.amask2=copy.deepcopy(m)
        fmedian=torch.median(f)
        mmedian=torch.median(m)
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if self.amask1[i][j]<=fmedian:
                    self.amask1[i][j]=0
                else:
                    self.amask1[i][j]=1
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if self.amask2[i][j]<=mmedian:
                    self.amask2[i][j]=0
                else:
                    self.amask2[i][j]=1
        return True

def DJVerify(AE,V):
    a,b=V.connect(AE)
    return float(a),float(b)

def MaskAE23(AE,Q):
    for param in AE.parameters():
        param.requires_grad=False
    MAE=copy.deepcopy(AE)
    MAE=MAE.to(device)
    p=MAE.state_dict()
    Q=Q.to(device)
    Qi=Q.inverse()
    p["fc2.weight"]=Q@p["fc2.weight"]
    p["fc3.weight"]=p["fc3.weight"]@Qi
    MAE.load_state_dict(p)
    MAE=MAE.to(device)
    return MAE

def MaskAE12(AE,Q):
    for param in AE.parameters():
        param.requires_grad=False
    MAE=copy.deepcopy(AE)
    MAE=MAE.to(device)
    p=MAE.state_dict()
    Q=Q.to(device)
    Qi=Q.inverse()
    p["fc1.weight"]=Q@p["fc1.weight"]
    p["fc2.weight"]=p["fc2.weight"]@Qi
    MAE.load_state_dict(p)
    MAE=MAE.to(device)
    return MAE

def GenQ(dim):
    Q=torch.eye(dim)
    for i in range(50*dim):
        temp=torch.randint(0,dim,(1,3))
        a=temp[0][0]
        b=temp[0][1]
        c=temp[0][2]
        q=torch.eye(dim)
        q[a][a]=0
        q[b][b]=0
        q[a][b]=1
        q[b][a]=1
        Q=q@Q
        alpha=torch.rand(size=(1,1))
        alpha=max(0.1,1+0.25*alpha[0][0])
        q=torch.eye(dim)
        q[c][c]=alpha
        Q=q@Q
    return Q

def GreedyPhiRow(W1,W2,verbose=True):
    if W1.shape!=W2.shape:
        return False
    if verbose:
        print("GreedyPhiRow Begins, Shape="+str(W1.shape))
    p=torch.eye(W1.shape[0])
    W3=copy.deepcopy(W2)
    for r in range(W1.shape[0]):
        mini=10000
        for r_ in range(r,W3.shape[0]):
            beta=(W1[r]@(W3[r_].t()))/(W3[r_]@(W3[r_].t()))
            diff=torch.sum((W1[r]-beta*W3[r_])**2)
            if diff<mini:
                mini=diff
                mini_r=r_
                mini_beta=beta
        pr=torch.eye(W1.shape[0])
        pr[r][r]=0
        pr[mini_r][mini_r]=0
        pr[mini_r][r]=1
        pr[r][mini_r]=mini_beta
        W3=pr@W3
        p=pr@p
        if verbose:
            print("Finishing index %i in %i, beta=%f, err=%f"%(r+1,W1.shape[0],mini_beta,mini))
    return p

def GreedyPhiColumn(W1,W2,verbose=True):
    if W1.shape!=W2.shape:
        return False
    if verbose:
        print("GreedyPhiColumn Begins, Shape="+str(W1.shape))
    W_1=copy.deepcopy(W1)
    W_2=copy.deepcopy(W2)
    W_1=W_1.t()
    W_2=W_2.t()
    return GreedyPhiRow(W_1,W_2,verbose)

def NM12(AEo,AEp,Z,device):
    AEo=AEo.to(device)
    AEp=AEp.to(device)
    Z=Z.to(device)
    M1o=AEo.first(Z)
    M1p=AEp.first(Z)
    M1o=M1o.detach().cpu()
    M1p=M1p.detach().cpu()
    Q=GreedyPhiColumn(M1o,M1p,False)
    AEc=MaskAE12(AEp,Q)
    return AEc

def NM23(AEo,AEp,Z,device):
    AEo=AEo.to(device)
    AEp=AEp.to(device)
    Z=Z.to(device)
    M1o=AEo.mid(Z)
    M1p=AEp.mid(Z)
    M1o=M1o.detach().cpu()
    M1p=M1p.detach().cpu()
    Q=GreedyPhiColumn(M1o,M1p,False)
    AEc=MaskAE23(AEp,Q)
    return AEc

bs=128

data_root="./data"
train_loader=torch.utils.data.DataLoader( 
            datasets.MNIST(data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32,32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=bs, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader( 
            datasets.MNIST(data_root, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((32,32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=bs,shuffle=True,num_workers=2)
print("MNIST Data loaded.")

E=10
AE1=MyAE()
#AE1.load_state_dict(torch.load("checkpoint/mnistAE2.pt"))
device=torch.device("cuda:1")
AE1=AE1.to(device)
for epoch in range(E):
    train(AE1,train_loader,device,True)
    torch.save(AE1.state_dict(),"checkpoint/mnistAE2.pt")

AE1=AE1.to(device)
Q3=GenQ(328)
AE3=MaskAE12(AE1,Q3)
Q4=GenQ(75)
AE4=MaskAE23(AE1,Q4)
Q51=GenQ(328)
Q52=GenQ(75)
AE5t=MaskAE12(AE1,Q51)
AE5=MaskAE23(AE5t,Q52)

# Simplified version. 
V2=UCVerifier()
V2.fit(AE1)
T=torch.randn(size=(100,1024))
V3=DSVerifier(T)
V3.fit(AE1)
V4=DJVerifier(T)
V4.fit(AE1)
Z=torch.randn(size=(100,1024))

print(DSVerify(AE1,V3))
print(DSVerify(AE3,V3))
print(DSVerify(AE4,V3))
print(DSVerify(AE5,V3))
print(DSVerify(NM12(AE1,NM23(AE1,AE3,Z,device),Z,device),V3))
print(DSVerify(NM12(AE1,NM23(AE1,AE4,Z,device),Z,device),V3))
print(DSVerify(NM12(AE1,NM23(AE1,AE5,Z,device),Z,device),V3))
