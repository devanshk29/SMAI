import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 

class CnnAutoencoder(nn.Module):
    def __init__(self,latent_dim=3,num_filters=32, kernel_size=3, lr=0.001, optimizer='adam'):
        super(CnnAutoencoder, self).__init__()
        # Encoder: Reduces spatial dimensions
        self.latent_dim = latent_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.lr = lr
        self.optimizer = optimizer
        self.encoder = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters*2, kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*num_filters*2, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 7*7*num_filters*2),
            nn.ReLU(),
            nn.Unflatten(1, (num_filters*2, 7, 7)),
            nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, 1, kernel_size, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )


        if self.optimizer == 'adam':
            self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)  # Adam optimizer
        elif self.optimizer == 'sgd':
            self.optim = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer")
        
       

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed
