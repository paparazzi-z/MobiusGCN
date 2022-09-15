from __future__ import absolute_import, division

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Computation of complex matrix in networks
def complex_compute(mat1, mat2):
    
    if mat1.dim() < 3 or (mat1.dim() == 3 and mat1.size(0) > 2):
        mat1_real = mat1
        mat1_imag = torch.zeros_like(mat1)
    else:
        mat1_real = mat1[0]
        mat1_imag = mat1[1]
    
    if mat2.dim() < 3 or (mat2.dim() == 3 and mat2.size(0) > 2):
        mat2_real = mat2
        mat2_imag = torch.zeros_like(mat2)
    else:
        mat2_real = mat2[0]
        mat2_imag = mat2[1]
    
    out_real = torch.matmul(mat1_real, mat2_real) - torch.matmul(mat1_imag, mat2_imag)
    out_imag = torch.matmul(mat1_imag, mat2_real) + torch.matmul(mat1_real, mat2_imag)
    out = torch.stack((out_real, out_imag), 0)
    
    return out

# Initialize a b c d randomly
def initialize_abcd(mat):
    a = np.zeros(16, dtype = complex)
    b = np.zeros(16, dtype = complex)
    c = np.zeros(16, dtype = complex)
    d = np.zeros(16, dtype = complex)
    
    from random import random
    for i in range(16):
        zp = [complex(mat[i]*random(),mat[i]*random()),
              complex(mat[i]*random(),mat[i]*random()),
              complex(mat[i]*random(),mat[i]*random())]
        wa = [complex(mat[i]*random(),mat[i]*random()),
              complex(mat[i]*random(),mat[i]*random()),
              complex(mat[i]*random(),mat[i]*random())]
        
        a[i] = np.linalg.det([[zp[0]*wa[0], wa[0], 1],
                              [zp[1]*wa[1], wa[1], 1],
                              [zp[2]*wa[2], wa[2], 1]])
        
        b[i] = np.linalg.det([[zp[0]*wa[0], zp[0], wa[0]],
                              [zp[1]*wa[1], zp[1], wa[1]], 
                              [zp[2]*wa[2], zp[2], wa[2]]])
        
        c[i] = np.linalg.det([[zp[0], wa[0], 1], 
                              [zp[1], wa[1], 1], 
                              [zp[2], wa[2], 1]])
        
        d[i] = np.linalg.det([[zp[0]*wa[0], zp[0], 1],
                              [zp[1]*wa[1], zp[1], 1],
                              [zp[2]*wa[2], zp[2], 1]])

    a_real = torch.tensor(np.float32(np.real(a)), dtype=torch.float)
    a_imag = torch.tensor(np.float32(np.imag(a)), dtype=torch.float)
    a_out = torch.stack((a_real, a_imag), 0)
    
    b_real = torch.tensor(np.float32(np.real(b)), dtype=torch.float)
    b_imag = torch.tensor(np.float32(np.imag(b)), dtype=torch.float)
    b_out = torch.stack((b_real, b_imag), 0)
    
    c_real = torch.tensor(np.float32(np.real(c)), dtype=torch.float)
    c_imag = torch.tensor(np.float32(np.imag(c)), dtype=torch.float)
    c_out = torch.stack((c_real, c_imag), 0)
    
    d_real = torch.tensor(np.float32(np.real(d)), dtype=torch.float)
    d_imag = torch.tensor(np.float32(np.imag(d)), dtype=torch.float)
    d_out = torch.stack((d_real, d_imag), 0)
    
    return a_out, b_out, c_out, d_out

    

class MobiusGraphConv(nn.Module):
    """
    Mobius graph convolution layer
    """

    def __init__(self, in_features, out_features, eigenVal, eigenVec, bias=True): 
        super(MobiusGraphConv, self).__init__()
        self.in_features = in_features # Input dimension
        self.out_features = out_features # Output dimension

        self.W = nn.Parameter(torch.empty(size=(2, in_features, out_features), dtype=torch.float)) # Transformation matrix
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.eigenVal = eigenVal
        self.eigenVec = eigenVec
        
#       Different methods to initialize a b c d        
        
        # pre-setted a b c d(not parameter in network)
#         self.A, self.B, self.C, self.D = initialize_abcd(self.eigenVal)
###################################################################################  
        # pre-setted a b c d and set as parameter
#         AA, BB, CC, DD = initialize_abcd(self.eigenVal)
        
#         self.A = nn.Parameter(AA)
#         self.B = nn.Parameter(BB)
#         self.C = nn.Parameter(CC)
#         self.D = nn.Parameter(DD)
###################################################################################  
        # randomly initialized parameter
#         self.A = nn.Parameter(torch.randn(size = (2,16), dtype=torch.float))
#         self.B = nn.Parameter(torch.randn(size = (2,16), dtype=torch.float))
#         self.C = nn.Parameter(torch.randn(size = (2,16), dtype=torch.float))
#         self.D = nn.Parameter(torch.randn(size = (2,16), dtype=torch.float))
###################################################################################      
        # use xavier_uniform_ to initialize parameter
        self.A = nn.Parameter(torch.empty(size = (2,16), dtype=torch.float))
        self.B = nn.Parameter(torch.empty(size = (2,16), dtype=torch.float))
        self.C = nn.Parameter(torch.empty(size = (2,16), dtype=torch.float))
        self.D = nn.Parameter(torch.empty(size = (2,16), dtype=torch.float))
        nn.init.xavier_uniform_(self.A.data, gain=1.414)
        nn.init.xavier_uniform_(self.B.data, gain=1.414)
        nn.init.xavier_uniform_(self.C.data, gain=1.414)
        nn.init.xavier_uniform_(self.D.data, gain=1.414)
#         print(self.A)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # convert eigenvalues to diagonal matrix
        eigenVal = torch.diag(self.eigenVal).to(input.device)
        eigenVec = self.eigenVec.to(input.device)
        
        # convert a b c d to diagonal matrix
        a = torch.stack((torch.diag(self.A[0]), torch.diag(self.A[1])), 0).to(input.device)
        b = torch.stack((torch.diag(self.B[0]), torch.diag(self.B[1])), 0).to(input.device)
        c = torch.stack((torch.diag(self.C[0]), torch.diag(self.C[1])), 0).to(input.device)
        d = torch.stack((torch.diag(self.D[0]), torch.diag(self.D[1])), 0).to(input.device)
#         print(a)
        
        m1 = torch.add(complex_compute(a, eigenVal), b)
#         print('m1')
#         print(m1)
        m2 = torch.inverse(torch.add(complex_compute(c, eigenVal), d))
#         print('m2')
#         print(m2)

#       Different orders to calculate mobius

        # calculate mobius transformation first
        M = complex_compute(m1, m2)
        r2 = complex_compute(eigenVec, M)
#################################################################
        # calculate in order
#         r1 = complex_compute(eigenVec, m1)
#         r2 = complex_compute(r1, m2)
        R = complex_compute(r2, torch.transpose(eigenVec, 0, 1))
        
        # Product of the transformation matrix and node reprensentations matrix
#         print(input.size())
        h = complex_compute(R, input)
        Z = complex_compute(h, self.W)
#         print('Z')
#         print(Z)
        
        output = 2*Z[0]
#         print('output')
#         print(output)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
