import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# initial varphi for rgb input
#init_varphi = np.asarray([[0.294, 0.110, 0.894],
#                          [0.750, 0.088, 0.425]])

# # initial varphi for bgr input
init_varphi = np.asarray([[0.6060, 1.2680, 0.7989],
                           [1.2383, 1.2540, 0.3927]])

class ACDModel(nn.Module):
    def __init__(self, input_dim, lambda_p=0.002, lambda_b=10, lambda_e=1, eta=0.6, gamma=0.5):
        super(ACDModel, self).__init__()
        self.lambda_p = lambda_p
        self.lambda_b = lambda_b
        self.lambda_e = lambda_e
        self.eta = eta
        self.gamma = gamma

        # Variables
        self.alpha = nn.Parameter(torch.tensor(init_varphi[0], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_varphi[1], dtype=torch.float32))
        self.w1= nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
        self.w3 = torch.tensor(1)
        self.w = [self.w1,self.w2,self.w3]
        #self.w = [nn.Parameter(torch.tensor(1.0,dtype=torch.float32)),nn.Parameter(torch.tensor(1.0,dtype=torch.float32)),torch.tensor(1)]
        #self.w = [nn.Parameter(torch.tensor(1.0, dtype=torch.float32)), nn.Parameter(torch.tensor(1.0, dtype=torch.float32)), torch.tensor(1.0)]

    def forward(self, input_od):
        sca_mat = torch.stack((torch.cos(self.alpha) * torch.sin(self.beta), torch.cos(self.alpha) * torch.cos(self.beta), torch.sin(self.alpha)), dim=1)
        cd_mat = torch.inverse(sca_mat)

        s = torch.matmul(input_od, cd_mat) * torch.stack(self.w)
        h, e, b = torch.split(s, 1, dim=1)

        l_p1 = torch.mean(b ** 2)
        l_p2 = torch.mean(2 * h * e / (h ** 2 + e ** 2))
        l_b = ((1 - self.eta) * torch.mean(h) - self.eta * torch.mean(e)) ** 2
        l_e = (self.gamma - torch.mean(s)) ** 2

        objective = l_p1 + self.lambda_p * l_p2 + self.lambda_b * l_b + self.lambda_e * l_e
        return objective, cd_mat, self.w
    

def acd_model(input_od, lambda_p=0.002, lambda_b=10, lambda_e=1, eta=0.6, gamma=0.5):
    model = ACDModel(input_od.shape[1], lambda_p, lambda_b, lambda_e, eta, gamma)
    optimizer = optim.Adagrad(model.parameters(), lr=0.05)

    # Forward pass
    objective, cd_mat, w = model(input_od)

    # Backward pass and optimization
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()

    return objective, cd_mat, w

"""
def acd_model(input_od, lambda_p=0.002, lambda_b=10, lambda_e=1, eta=0.6, gamma=0.5):
    #Stain matrix estimation by
    #"Yushan Zheng, et al., Adaptive Color Deconvolution for Histological WSI Normalization."


    alpha = torch.tensor(init_varphi[0], dtype=torch.float32)
    beta = torch.tensor(init_varphi[1], dtype=torch.float32)
    # Create tensors for trainable weights
    w1 = torch.tensor(1.0)
    w2 = torch.tensor(1.0)

    # Create a constant tensor (not trainable)
    w3 = torch.tensor(1.0)

    # Combine them into a list
    w = [w1,w2,w3]

    sca_mat = torch.stack((torch.cos(alpha) * torch.sin(beta), torch.cos(alpha) * torch.cos(beta), torch.sin(alpha)), dim=1)
    cd_mat = torch.inverse(sca_mat)

    s = torch.matmul(input_od, cd_mat) * torch.stack(w)
    h, e, b = torch.split(s, 1, dim=1)

    l_p1 = torch.mean(b ** 2)
    l_p2 = torch.mean(2 * h * e / (h ** 2 + e ** 2))
    l_b = ((1 - eta) * torch.mean(h) - eta * torch.mean(e)) ** 2
    l_e = (gamma - torch.mean(s)) ** 2

    objective = l_p1 + lambda_p * l_p2 + lambda_b * l_b + lambda_e * l_e
    
    optimizer = torch.optim.Adagrad([objective, cd_mat, w], lr=0.05)
    
    optimizer.zero_grad()  # Zero out gradients from previous step
    objective.backward()  # Calculate gradients
    optimizer.step()  # Update weights based on gradients

    return target, cd_mat, w
    
"""
