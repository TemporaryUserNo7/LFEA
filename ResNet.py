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
import resnet_8x
import time

class BasicBlock(nn.Module):
    expansion=1 
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes) 
        self.shortcut = nn.Sequential()
        if stride!=1 or in_planes!=self.expansion*planes:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)) 
    def forward(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out
 
class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes=64
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.layer1=self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2=self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3=self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4=self._make_layer(block,512,num_blocks[3],stride=2)
        self.linear=nn.Linear(512*block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    def _make_layer(self,block,planes,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes=planes*block.expansion
        return nn.Sequential(*layers)
    def forward(self,x,out_feature=False):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=F.avg_pool2d(out,4)
        feature=out.view(out.size(0),-1)
        out=self.linear(feature)
        if out_feature==False:
            return out
        else:
            return out,feature
    def c2(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        for B in self.layer2.named_children():
            if B[0]=="0":
                out=B[1](out)
        for B in self.layer2.named_children():
            if B[0]=="1":
                out=B[1](out)
        for B in self.layer2.named_children():
            if B[0]=="2":
                BB=B[1]
        out=BB.bn1(BB.conv1(out))
        return out
    def c3(self,x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.layer1(out)
        out=self.layer2(out)
        for B in self.layer3.named_children():
            if B[0]=="0":
                out=B[1](out)
        for B in self.layer3.named_children():
            if B[0]=="1":
                out=B[1](out)
        for B in self.layer3.named_children():
            if B[0]=="2":
                out=B[1](out)
        for B in self.layer3.named_children():
            if B[0]=="3":
                BB=B[1]
        out=BB.bn1(BB.conv1(out))
        return out
    
def ResNet34_8x(num_classes=10):
    return ResNet(BasicBlock,[3,4,6,3],num_classes)

def myprune(model,rate,device):
    local_model=copy.deepcopy(model)
    local_model=local_model.to(device)
    for m in local_model.named_modules():
        if m[0]=="layer2.2.conv1" or m[0]=="layer3.3.conv1" or m[0]=="layer4.1.conv1" or m[0]=="linear" or m[0]=="layer1.0.conv1":
            module=m[1]
            prune.random_unstructured(module,name="weight",amount=rate)
            prune.remove(module,"weight")
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
    CNN=CNN.to(device)
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

def freeze(CNN):
    for param in CNN.parameters():
        param.requires_grad_=False
    for mo in CNN.modules():
        if isinstance(mo,torch.nn.modules.batchnorm.BatchNorm2d):
            mo.track_running_stats=False
    return True

class UCVerifier(nn.Module):
    def __init__(self,r=0.1):
        super(UCVerifier,self).__init__()
        self.r=r
        self.mask1=torch.zeros(size=(128,128,3,3))
        self.mask2=torch.zeros(size=(256,256,3,3))
        self.mask1.requires_grad=False
        self.mask2.requires_grad=False
        for i in range(128):
            for j in range(128):
                for k in range(3):
                    for l in range(3):
                        if torch.rand(size=(1,1))[0][0]<=r:
                            self.mask1[i][j][k][l]=1
        for i in range(256):
            for j in range(256):
                for k in range(3):
                    for l in range(3):
                        if torch.rand(size=(1,1))[0][0]<=r:
                            self.mask2[i][j][k][l]=1
        self.carrier1=torch.zeros(size=(128,128,3,3))
        self.carrier2=torch.zeros(size=(256,256,3,3))
    def connect(self,CNN):
        c1=CNN.state_dict()["layer2.2.conv1.weight"].detach().cpu()
        c2=CNN.state_dict()["layer3.3.conv1.weight"].detach().cpu()
        e1=(c1-self.carrier1)*self.mask1
        e2=(c2-self.carrier2)*self.mask2
        return torch.sum(e1**2)+torch.sum(e2**2)
    def fit(self,CNN):
        c1=CNN.state_dict()["layer2.2.conv1.weight"].detach().cpu()
        c2=CNN.state_dict()["layer3.3.conv1.weight"].detach().cpu()
        self.carrier1=copy.deepcopy(c1)
        self.carrier2=copy.deepcopy(c2)
        return True
    def test(self,CNN,tau=0.001):
        e=self.connect(CNN)
        ae=e/(737280*self.r)
        return float(ae)

class DSVerifier(nn.Module):
    def __init__(self,T):
        super(DSVerifier,self).__init__()
        self.T=T
        self.mask1=torch.zeros(size=(100,128))
        self.mask2=torch.zeros(size=(100,256))
        self.median1=0
        self.median2=0
    def connect(self,CNN):
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach()
        c3=c3.detach()
        tm1=torch.zeros(size=(100,128))
        tm2=torch.zeros(size=(100,256))
        #tm1=tm1.to(device)
        #tm2=tm2.to(device)
        for i in range(100):
            for j in range(128):
                if c2[i][j][7][7]<=self.median1:
                    tm1[i][j]=0
                else:
                    tm1[i][j]=1
        for i in range(100):
            for j in range(256):
                if c3[i][j][3][3]<=self.median2:
                    tm2[i][j]=0
                else:
                    tm2[i][j]=1
        return torch.sum((tm1-self.mask1)**2)+torch.sum((tm2-self.mask2)**2)
    def fit(self,CNN):
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach()
        c3=c3.detach()
        tm1=torch.zeros(size=(100,128))
        tm2=torch.zeros(size=(100,256))
        tm1=tm1.to(device)
        tm2=tm2.to(device)
        for i in range(100):
            for j in range(128):
                tm1[i][j]=c2[i][j][7][7]
        for i in range(100):
            for j in range(256):
                tm2[i][j]=c3[i][j][3][3]
        self.median1=torch.median(tm1)
        self.median2=torch.median(tm2)
        for i in range(100):
            for j in range(128):
                if tm1[i][j]<=self.median1:
                    self.mask1[i][j]=0
                else:
                    self.mask1[i][j]=1
        for i in range(100):
            for j in range(256):
                if tm2[i][j]<=self.median2:
                    self.mask2[i][j]=0
                else:
                    self.mask2[i][j]=1
        return True
    def test(self,CNN,tau=0.1):
        e=self.connect(CNN)
        e=e/(12800+25600)
        return float(e)

class DJVerifier(nn.Module):
    def __init__(self,T):
        super(DJVerifier,self).__init__()
        self.T=T
        self.vmask1=torch.zeros(size=(100,128))
        self.vmask2=torch.zeros(size=(100,256))
        self.amask1=torch.zeros(size=(100,128))
        self.amask2=torch.zeros(size=(100,256))
    def connect(self,CNN):
        CNN=CNN.to(device)
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach().cpu()
        c3=c3.detach().cpu()
        tm1=torch.zeros(size=(100,128))
        tm2=torch.zeros(size=(100,256))
        tm1=tm1.to(device)
        tm2=tm2.to(device)
        for i in range(100):
            for j in range(128):
                tm1[i][j]=c2[i][j][7][7]
        for i in range(100):
            for j in range(256):
                tm2[i][j]=c3[i][j][3][3]
        p=torch.norm(tm1-self.vmask1,2)+torch.norm(tm2-self.vmask2,2)
        p=p.detach().cpu()
        p=float(p)/(12800+25600)
        c2m=torch.median(tm1)
        c3m=torch.median(tm2)
        for i in range(100):
            for j in range(128):
                if tm1[i][j]<c2m:
                    tm1[i][j]=0
                else:
                    tm1[i][j]=1
        for i in range(100):
            for j in range(256):
                if tm2[i][j]<c3m:
                    tm2[i][j]=0
                else:
                    tm2[i][j]=1
        q=torch.norm(tm1-self.amask1,2)+torch.norm(tm2-self.amask2,2)
        q=q.detach().cpu()
        q=float(q)/(128+256)
        return p,q

    def fit(self,CNN):
        c2=CNN.c2(self.T)
        c3=CNN.c3(self.T)
        c2=c2.detach().cpu()
        c3=c3.detach().cpu()
        tm1=torch.zeros(size=(100,128))
        tm2=torch.zeros(size=(100,256))
        tm1=tm1.to(device)
        tm2=tm2.to(device)
        for i in range(100):
            for j in range(128):
                tm1[i][j]=c2[i][j][7][7]
        for i in range(100):
            for j in range(256):
                tm2[i][j]=c3[i][j][3][3]
        self.vmask1=copy.deepcopy(tm1)
        self.vmask2=copy.deepcopy(tm2)
        c2m=torch.median(tm1)
        c3m=torch.median(tm2)
        for i in range(100):
            for j in range(128):
                if tm1[i][j]<c2m:
                    self.amask1[i][j]=0
                else:
                    self.amask1[i][j]=1
        for i in range(100):
            for j in range(256):
                if tm2[i][j]<c3m:
                    self.amask2[i][j]=0
                else:
                    self.amask2[i][j]=1
        self.amask1=self.amask1.to(device)
        self.amask2=self.amask2.to(device)
        return True
    def test(self,CNN):
        return self.connect(CNN)

def GenQ(dim):
    Q=torch.eye(dim)
    Q2=torch.eye(dim)
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
        Q2=q@Q2
        alpha=torch.rand(size=(1,1))
        alpha=max(0.1,1+0.25*alpha[0][0])
        q=torch.eye(dim)
        q[c][c]=alpha
        Q=q@Q
        Q2=(q@q)@Q2
    return Q,Q2

# Q@C
def ConvMulRight(Q,C):
    if not(Q.dim()==2 and C.dim()==4 and Q.shape[0]==Q.shape[1] and Q.shape[0]==C.shape[0]):
        print("Shape mismatch.")
        return False
    C_=C.detach()
    ans=copy.deepcopy(C_)
    ans=ans.to(device)
    for i in range(C.shape[0]):
        print("CMR,%i in %i"%(i,C.shape[0]))
        for j in range(C.shape[1]):
            temp=torch.zeros(size=(C.shape[2],C.shape[3]))
            temp=temp.to(device)
            for k in range(Q.shape[0]):
                if Q[i][k]!=0:
                    temp=temp+Q[i][k]*C_[k][j]
            ans[i][j]=temp
    return ans

# C@Q
def ConvMulLeft(Q,C):
    if not(Q.dim()==2 and C.dim()==4 and Q.shape[0]==Q.shape[1] and Q.shape[0]==C.shape[1]):
        print("Shape mismatch.")
        return False
    C_=C.detach()
    ans=copy.deepcopy(C_)
    ans=ans.to(device)
    for i in range(C.shape[0]):
        print("CML,%i in %i"%(i,C.shape[0]))
        for j in range(C.shape[1]):
            temp=torch.zeros(size=(C.shape[2],C.shape[3]))
            temp=temp.to(device)
            for k in range(Q.shape[0]):
                if Q[k][j]!=0:
                    temp=temp+C_[i][k]*Q[k][j]
            ans[i][j]=temp
    return ans

def MaskBB2(CNN,Q,Q2):
    Q=Q.to(device)
    Q2=Q2.to(device)
    d=CNN.state_dict()
    localCNN=ResNet34_8x(10)
    localCNN=localCNN.to(device)
    localDic=copy.deepcopy(d)
    localDic["layer2.2.conv1.weight"]=ConvMulRight(Q,CNN.state_dict()["layer2.2.conv1.weight"])
    localDic["layer2.2.bn1.running_mean"]=Q@CNN.state_dict()["layer2.2.bn1.running_mean"]
    localDic["layer2.2.bn1.running_var"]=Q2@CNN.state_dict()["layer2.2.bn1.running_var"]
    localDic["layer2.2.bn1.weight"]=Q@CNN.state_dict()["layer2.2.bn1.weight"]
    localDic["layer2.2.bn1.bias"]=Q@CNN.state_dict()["layer2.2.bn1.bias"]
    localDic["layer2.2.conv2.weight"]=ConvMulLeft(Q.inverse(),CNN.state_dict()["layer2.2.conv2.weight"])
    localCNN.load_state_dict(localDic)
    localCNN.eval()
    freeze(localCNN)
    return localCNN

def MaskBB3(CNN,Q,Q2):
    Q=Q.to(device)
    Q2=Q2.to(device)
    d=CNN.state_dict()
    localCNN=ResNet34_8x(10)
    localCNN=localCNN.to(device)
    localDic=copy.deepcopy(d)
    localDic["layer3.3.conv1.weight"]=ConvMulRight(Q,CNN.state_dict()["layer3.3.conv1.weight"])
    localDic["layer3.3.bn1.running_mean"]=Q@CNN.state_dict()["layer3.3.bn1.running_mean"]
    localDic["layer3.3.bn1.running_var"]=Q2@CNN.state_dict()["layer3.3.bn1.running_var"]
    localDic["layer3.3.bn1.weight"]=Q@CNN.state_dict()["layer3.3.bn1.weight"]
    localDic["layer3.3.bn1.bias"]=Q@CNN.state_dict()["layer3.3.bn1.bias"]
    localDic["layer3.3.conv2.weight"]=ConvMulLeft(Q.inverse(),CNN.state_dict()["layer3.3.conv2.weight"])
    localCNN.load_state_dict(localDic)
    localCNN.eval()
    freeze(localCNN)
    return localCNN

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
    CNNo=CNNo.to(device)
    CNNp=CNNp.to(device)
    Z=Z.to(device)
    M1o=CNNo.c3(Z)
    M1p=CNNp.c3(Z)
    M1o=M1o.to(torch.device("cpu"))
    M1p=M1p.to(torch.device("cpu"))
    tempo=torch.zeros(size=(Z.shape[0],256))
    tempp=torch.zeros(size=(Z.shape[0],256))
    # Point calibration.
    for i in range(Z.shape[0]):
        for j in range(256):
            tempo[i][j]=M1o[i][j][2][2]
            tempp[i][j]=M1p[i][j][2][2]
    tempo=tempo.detach().cpu()
    tempp=tempp.detach().cpu()
    Q=GreedyPhiColumn(tempo,tempp,False)
    Q2=Q**2
    CNNc=MaskBB3(CNNp,Q,Q2)
    return CNNc

def NMConv2(CNNo,CNNp,Z):
    CNNo=CNNo.to(device)
    CNNp=CNNp.to(device)
    Z=Z.to(device)
    M1o=CNNo.c2(Z)
    M1p=CNNp.c2(Z)
    M1o=M1o.to(torch.device("cpu"))
    M1p=M1p.to(torch.device("cpu"))
    tempo=torch.zeros(size=(Z.shape[0],128))
    tempp=torch.zeros(size=(Z.shape[0],128))
    # Point calibration.
    for i in range(Z.shape[0]):
        for j in range(128):
            tempo[i][j]=M1o[i][j][5][5]
            tempp[i][j]=M1p[i][j][5][5]
    tempo=tempo.detach().cpu()
    tempp=tempp.detach().cpu()
    Q=GreedyPhiColumn(tempo,tempp,False)
    Q2=Q**2
    CNNc=MaskBB2(CNNp,Q,Q2)
    return CNNc

bs=128

data_root="./data"
train_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(data_root, train=True, download=True,
                       transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=bs, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader( 
            datasets.CIFAR10(data_root, train=False, download=True,
                       transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])),
            batch_size=bs, shuffle=True, num_workers=2)

CNN1=ResNet34_8x(10)
device=torch.device("cuda:1")
CNN1=CNN1.to(device)
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

CNN1.eval()
freeze(CNN1)

#Q3,Q3_=GenQ(128)
#CNN3=MaskBB2(CNN1,Q3,Q3_)
#Q4,Q4_=GenQ(256)
#CNN4=MaskBB3(CNN1,Q4,Q4_)
Q5,Q5_=GenQ(128)
Q5e,Q5e_=GenQ(256)
CNN5=MaskBB3(MaskBB2(CNN1,Q5,Q5_),Q5e,Q5e_)
#freeze(CNN3)
#freeze(CNN4)
freeze(CNN5)

Z=torch.randn(size=(100,3,32,32))

#CNN7=NMConv2(CNN1,NMConv3(CNN1,CNN3,Z),Z)
#CNN8=NMConv2(CNN1,NMConv3(CNN1,CNN4,Z),Z)
CNN9=NMConv2(CNN1,NMConv3(CNN1,CNN5,Z),Z)

V2=UCVerifier()
V2.fit(CNN1)
T1=torch.randn(size=(100,3,32,32))
T1=T1.to(device)
CNN1=CNN1.to(device)
V3=DSVerifier(T1)
V3.fit(CNN1)
T2=torch.randn(size=(100,3,32,32))
T2=T2.to(device)
V4=DJVerifier(T2)
V4.fit(CNN1)

print(V3.test(CNN1))
#print(V3.test(CNN3))
#print(V3.test(CNN4))
print(V3.test(CNN5))
#print(V3.test(CNN7))
#print(V3.test(CNN8))
print(V3.test(CNN9))
