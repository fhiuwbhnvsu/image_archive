import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def build_otflow(data_dim):
    return OTFlow(data_dim)

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.input_dim = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """

        x = self.act(self.layers[0].forward(x))

        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x

class Phi(nn.Module):
    def __init__(self, nTh, m, d, use_t=True, r=10):
        """
            neural network approximating Phi (see Eq. (9) in our paper)

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:   int, number of resNet layers , (number of theta layers)
        :param m:     int, hidden dimension
        :param d:     int, dimension of space input (expect inputs to be d+1 for space-time)
        :param use_t: int, use time parameter or not
        :param r:    int, rank r for the A matrix
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.input_dim    = d
        self.use_t = use_t

        r = min(r,d+int(self.use_t)) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(r, d+int(self.use_t)) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+int(self.use_t), 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.N = ResNN(d+int(self.use_t), m, nTh=nTh)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def forward(self, t, x):
        """ calculating Phi(s, theta)...not used in OT-Flow """
        if self.use_t:
            x = F.pad(x, (0, 1, 0, 0), value=t)
        # force A to be symmetric
        symA = torch.matmul(torch.t(self.A), self.A) # A'A
        return self.w( self.N(x)) + 0.5 * torch.sum( torch.matmul(x , symA) * x , dim=1, keepdims=True) + self.c(x)

    def grad(self, xt):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi
        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh    
        x = xt

        N    = self.N
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = torch.tanh(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , torch.tanh( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + torch.mm(symA, x.t() ) + self.c.weight.t()
        return grad.t()


class OTFlow(nn.Module):
    def __init__(self, input_dim):
        super(OTFlow, self).__init__()
        self.d = input_dim
        self.phi = Phi(nTh=2, m=32, d=self.d)
    
    def forward(self, x, t):
        xt = torch.cat((x, t.unsqueeze(1)), dim=1)
        drift = -self.phi.grad(xt.float())
        return drift[:, :-1]