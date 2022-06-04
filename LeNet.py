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

class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet,self).__init__()
        self.conv1=nn.Conv2d(1,6,kernel_size=(5,5))
        self.relu1=nn.ReLU()
        self.maxpool1=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2=nn.Conv2d(6,16,kernel_size=(5,5))
        self.relu2=nn.ReLU()
        self.maxpool2=nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv3=nn.Conv2d(16,120,kernel_size=(5,5))
        self.relu3=nn.ReLU()
        self.fc1=nn.Linear(120,84)
        self.relu4=nn.ReLU()
        self.fc2=nn.Linear(84,10)
    def forward(self,img,out_feature=False):
        output=self.conv1(img)
        output=self.relu1(output)
        output=self.maxpool1(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.maxpool2(output)
        output=self.conv3(output)
        output=self.relu3(output)
        feature=output.view(-1, 120)
        output=self.fc1(feature)
        output=self.relu4(output)
        output=self.fc2(output)
        if out_feature==False:
            return output
        else:
            return output,feature
    def c2(self,img):
        output=self.conv1(img)
        output=self.relu1(output)
        output=self.maxpool1(output)
        output=self.conv2(output)
        return output
    def c3(self,img):
        output=self.conv1(img)
        output=self.relu1(output)
        output=self.maxpool1(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.maxpool2(output)
        output=self.conv3(output)
        return output

def myprune(model,rate,device):
    local_model=copy.deepcopy(model)
    local_model=local_model.to(device)
    module1=local_model.conv2
    prune.random_unstructured(module1,name="weight",amount=rate)
    module2=local_model.conv3
    prune.random_unstructured(module2,name="weight",amount=rate)
    module3=local_model.fc1
    prune.random_unstructured(module3,name="weight",amount=rate)
    prune.remove(module1,"weight")
    prune.remove(module2,"weight")
    prune.remove(module3,"weight")
    return local_model

def train(CNN,train_loader,lr,device,verbose=True):
    for param in CNN.parameters():
        param.requires_grad=True
    optimizer=optim.Adam(CNN.parameters(),lr=lr)
    E=1
    lam=0.05
    for epoch in range(E):
        for idx,(b_x,b_y) in enumerate(train_loader):
            b_x=b_x.to(device)
            b_y=b_y.to(device)
            optimizer.zero_grad()
            pred=CNN(b_x)
            l=F.cross_entropy(pred,b_y)
            l.backward()
            optimizer.step()
            if idx%100==0 and verbose:
                print("Epoch=%i,Index=%i,Loss=%f"%(epoch,idx,float(l.detach())/bs))
    return True

def test(CNN,test_loader,device,verbose=True):
    CNN.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=CNN(data)
            test_loss+=F.cross_entropy(output,target,reduction='sum').item() # sum up batch loss
            pred=output.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        test_loss,correct,len(test_loader.dataset),
        100. * correct/len(test_loader.dataset)))
    return correct/len(test_loader.dataset)

class UCVerifier(nn.Module):
    def __init__(self,r=0.1):
        super(UCVerifier,self).__init__()
        self.r=r
        self.mask1=torch.zeros(size=(16,6,5,5))
        self.mask2=torch.zeros(size=(120,16,5,5))
        self.mask1.requires_grad=False
        self.mask2.requires_grad=False
        for i in range(16):
            for j in range(6):
                for k in range(5):
                    for l in range(5):
                        if torch.rand(size=(1,1))[0][0]<=r:
                            self.mask1[i][j][k][l]=1
        for i in range(120):
            for j in range(16):
                for k in range(5):
                    for l in range(5):
                        if torch.rand(size=(1,1))[0][0]<=r:
                            self.mask2[i][j][k][l]=1
        self.carrier1=torch.zeros(size=(16,6,5,5))
        self.carrier2=torch.zeros(size=(120,16,5,5))
    def connect(self,CNN):
        c1=CNN.state_dict()["conv2.weight"].detach().cpu()
        c2=CNN.state_dict()["conv3.weight"].detach().cpu()
        e1=(c1-self.carrier1)*self.mask1
        e2=(c2-self.carrier2)*self.mask2
        return float(torch.sum(e1**2)+torch.sum(e2**2))/(16*6*5*5+120*16*5*5)/0.1
    def fit(self,CNN):
        c1=CNN.state_dict()["conv2.weight"].detach().cpu()
        c2=CNN.state_dict()["conv3.weight"].detach().cpu()
        self.carrier1=copy.deepcopy(c1)
        self.carrier2=copy.deepcopy(c2)
        return True
    def test(self,CNN,tau=0.001):
        e=self.connect(CNN)
        return float(e)

class DSVerifier(nn.Module):
    def __init__(self,T):
        super(DSVerifier,self).__init__()
        self.T=T
        self.mask1=torch.zeros(size=(100,16,10,10))
        self.mask2=torch.zeros(size=(100,120,1,1))
        self.median1=0
        self.median2=0
    def connect(self,CNN):
        CNN=CNN.to(torch.device("cpu"))
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach()
        c3=c3.detach()
        for i in range(100):
            for j in range(16):
                for k in range(10):
                    for l in range(10):
                        if c2[i][j][k][l]<=self.median1:
                            c2[i][j][k][l]=0
                        else:
                            c2[i][j][k][l]=1
        for i in range(100):
            for j in range(120):
                for k in range(1):
                    for l in range(1):
                        if c3[i][j][k][l]<=self.median2:
                            c3[i][j][k][l]=0
                        else:
                            c3[i][j][k][l]=1
        return float(torch.sum((c2-self.mask1)**2)+torch.sum((c3-self.mask2)**2))/(160000+12000)
    def fit(self,CNN):
        CNN=CNN.to(torch.device("cpu"))
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach()
        c3=c3.detach()
        self.median1=torch.median(c2)
        self.median2=torch.median(c3)
        for i in range(100):
            for j in range(16):
                for k in range(10):
                    for l in range(10):
                        if c2[i][j][k][l]<=self.median1:
                            self.mask1[i][j][k][l]=0
                        else:
                            self.mask1[i][j][k][l]=1
        for i in range(100):
            for j in range(120):
                for k in range(1):
                    for l in range(1):
                        if c3[i][j][k][l]<=self.median2:
                            self.mask2[i][j][k][l]=0
                        else:
                            self.mask2[i][j][k][l]=1
        return True
    def test(self,CNN,tau=0.1):
        e=self.connect(CNN)
        return e

class DJVerifier(nn.Module):
    def __init__(self,T):
        super(DJVerifier,self).__init__()
        self.T=T
        self.vmask1=torch.zeros(size=(100,16,10,10))
        self.vmask2=torch.zeros(size=(100,120,1,1))
        self.amask1=torch.zeros(size=(100,16,10,10))
        self.amask2=torch.zeros(size=(100,120,1,1))
    def connect(self,CNN):
        CNN=CNN.to(torch.device("cpu"))
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach().cpu()
        c3=c3.detach().cpu()
        p=torch.norm(c2-self.vmask1,2)+torch.norm(c3-self.vmask2,2)
        p=p.detach().cpu()
        p=float(p)/(160000+12000)
        c2m=torch.median(c2)
        c3m=torch.median(c3)
        for i in range(100):
            for j in range(16):
                for k in range(10):
                    for l in range(10):
                        if c2[i][j][k][l]<c2m:
                            c2[i][j][k][l]=0
                        else:
                            c2[i][j][k][l]=1
        for i in range(100):
            for j in range(120):
                for k in range(1):
                    for l in range(1):
                        if c3[i][j][k][l]<c3m:
                            c3[i][j][k][l]=0
                        else:
                            c3[i][j][k][l]=1
        q=torch.norm(c2-self.amask1,2)+torch.norm(c3-self.amask2,2)
        q=q.detach().cpu()
        q=float(q)/(160000+12000)*200
        return p,q
    def fit(self,CNN):
        CNN=CNN.to(torch.device("cpu"))
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach().cpu()
        c3=c3.detach().cpu()
        self.vmask1=copy.deepcopy(c2)
        self.vmask2=copy.deepcopy(c3)
        c2m=torch.median(c2)
        c3m=torch.median(c3)
        for i in range(100):
            for j in range(16):
                for k in range(10):
                    for l in range(10):
                        if c2[i][j][k][l]<c2m:
                            self.amask1[i][j][k][l]=0
                        else:
                            self.amask1[i][j][k][l]=1
        for i in range(100):
            for j in range(120):
                for k in range(1):
                    for l in range(1):
                        if c3[i][j][k][l]<c3m:
                            self.amask2[i][j][k][l]=0
                        else:
                            self.amask2[i][j][k][l]=1
        return True
    def test(self,CNN):
        return self.connect(CNN)

def GenQ(dim):
    Q=torch.eye(dim)
    for i in range(10*dim):
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

# Q@C
def ConvMulRight(Q,C):
    if not(Q.dim()==2 and C.dim()==4 and Q.shape[0]==Q.shape[1] and Q.shape[0]==C.shape[0]):
        print("Shape mismatch.")
        return False
    C_=C.detach().cpu()
    ans=copy.deepcopy(C_)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            temp=torch.zeros(size=(C.shape[2],C.shape[3]))
            for k in range(Q.shape[0]):
                temp=temp+Q[i][k]*C_[k][j]
            ans[i][j]=temp
    return ans

# C@Q
def ConvMulLeft(Q,C):
    if not(Q.dim()==2 and C.dim()==4 and Q.shape[0]==Q.shape[1] and Q.shape[0]==C.shape[1]):
        print("Shape mismatch.")
        return False
    C_=C.detach().cpu()
    ans=copy.deepcopy(C_)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            temp=torch.zeros(size=(C.shape[2],C.shape[3]))
            for k in range(Q.shape[0]):
                temp=temp+C_[i][k]*Q[k][j]
            ans[i][j]=temp
    return ans

def MaskConv2(CNN,Q):
    Q=Q.to(torch.device("cpu"))
    CNN_=CNN.to(torch.device("cpu"))
    local_dict=CNN_.state_dict()
    W1=local_dict["conv2.weight"]
    B1=local_dict["conv2.bias"]
    W2=local_dict["conv3.weight"]
    W1_=ConvMulRight(Q,W1)
    B1_=Q@B1
    W2_=ConvMulLeft(Q.inverse(),W2)
    new_dict=copy.deepcopy(local_dict)
    new_dict["conv2.weight"]=copy.deepcopy(W1_)
    new_dict["conv2.bias"]=copy.deepcopy(B1_)
    new_dict["conv3.weight"]=copy.deepcopy(W2_)
    MaskedCNN=MyLeNet()
    MaskedCNN.load_state_dict(new_dict)
    return MaskedCNN

def MaskConv3(CNN,Q):
    Q=Q.to(torch.device("cpu"))
    CNN_=CNN.to(torch.device("cpu"))
    local_dict=CNN_.state_dict()
    W1=local_dict["conv3.weight"]
    B1=local_dict["conv3.bias"]
    W2=local_dict["fc1.weight"]
    W1_=ConvMulRight(Q,W1)
    B1_=Q@B1
    W2_=W2@Q.inverse()
    new_dict=copy.deepcopy(local_dict)
    new_dict["conv3.weight"]=copy.deepcopy(W1_)
    new_dict["conv3.bias"]=copy.deepcopy(B1_)
    new_dict["fc1.weight"]=copy.deepcopy(W2_)
    MaskedCNN=MyLeNet()
    MaskedCNN.load_state_dict(new_dict)
    return MaskedCNN

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

def NMConv3(CNNo,CNNp,Z):
    CNNo=CNNo.to(torch.device("cpu"))
    CNNp=CNNp.to(torch.device("cpu"))
    Z=Z.to(torch.device("cpu"))
    M1o=CNNo.c3(Z)
    M1p=CNNp.c3(Z)
    M1o=M1o.reshape(Z.shape[0],120)
    M1p=M1p.reshape(Z.shape[0],120)
    M1o=M1o.detach().cpu()
    M1p=M1p.detach().cpu()
    Q=GreedyPhiColumn(M1o,M1p,False)
    CNNc=MaskConv3(CNNp,Q)
    return CNNc

def NMConv2(CNNo,CNNp,Z):
    CNNo=CNNo.to(torch.device("cpu"))
    CNNp=CNNp.to(torch.device("cpu"))
    Z=Z.to(torch.device("cpu"))
    M1o=CNNo.c2(Z)
    M1p=CNNp.c2(Z)
    tempo=torch.zeros(size=(Z.shape[0],16))
    tempp=torch.zeros(size=(Z.shape[0],16))
    # Point calibration.
    for i in range(Z.shape[0]):
        for j in range(16):
            tempo[i][j]=M1o[i][j][5][5]
            tempp[i][j]=M1p[i][j][5][5]
    tempo=tempo.detach().cpu()
    tempp=tempp.detach().cpu()
    Q=GreedyPhiColumn(tempo,tempp,False)
    CNNc=MaskConv2(CNNp,Q)
    return CNNc

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

CNN1=MyLeNet()
lr=0.00005
for i in range(90):
    if i==20:
        lr=0.00001
    if i==50:
        lr=0.000005
    if i==80:
        lr=0.000002
    print("Epoch %i in %i."%(i+1,90))
    train(CNN1,train_loader,lr,device,False)
    test(CNN1,test_loader,device,True)

Q1=GenQ(16)
Q2=GenQ(120)
Q3=GenQ(16)
Q4=GenQ(120)
CNN3=MaskConv2(CNN1,Q1)
CNN4=MaskConv3(CNN1,Q2)
CNN5=MaskConv3(MaskConv2(CNN1,Q3),Q4)

Z=torch.randn(size=(100,1,32,32))
CNN7=NMConv2(CNN1,NMConv3(CNN1,CNN3,Z),Z)
CNN8=NMConv2(CNN1,NMConv3(CNN1,CNN4,Z),Z)
CNN9=NMConv2(CNN1,NMConv3(CNN1,CNN5,Z),Z)

V2=UCVerifier()
V2.fit(CNN1)
T1=torch.randn(size=(100,1,32,32))
V3=DSVerifier(T1)
V3.fit(CNN1)
T2=torch.randn(size=(100,1,32,32))
V4=DJVerifier(T2)
V4.fit(CNN1)

print(V3.test(CNN1))
print(V3.test(CNN3))
print(V3.test(CNN4))
print(V3.test(CNN5))
print(V3.test(CNN7))
print(V3.test(CNN8))
print(V3.test(CNN9))
