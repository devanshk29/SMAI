import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class CNN(nn.Module):
    def __init__(self, layer_config=[], input_channels=1, input_size=(128,128), num_classes=10, task='classification'):
        super().__init__()
        
        
        self.task = task
        self.activations = []          
        self.layers = nn.ModuleList()  
        
        current_size = input_size
        layer_in = input_channels

        for config in layer_config:
            if config['type'] == 'pool':
                if config['pool_type'] == 'max':
                    self.layers.append(nn.MaxPool2d(kernel_size=config['kernel_size'], stride=2))
                elif config['pool_type'] == 'avg':
                    self.layers.append(nn.AvgPool2d(kernel_size=config['kernel_size'], stride=2))
                else:
                    raise ValueError('Invalid pool type')

                current_size = self.calculate_pool_output_size(current_size, config['kernel_size'], stride=2)
            elif config['type'] == 'conv2d':
                layer_out = config['out_channels']
                self.layers.append(nn.Conv2d(layer_in, layer_out, kernel_size=config['kernel_size'], stride=config['stride'], padding=config['padding']))
                self.activations.append(self.get_activation(config['activation']))

                current_size = self.calculate_conv_output_size(current_size, config['kernel_size'], config['stride'], config['padding'])
                layer_in = layer_out  

            


        flattened_size = layer_in * current_size[0] * current_size[1]
        self.fc1 = nn.Linear(flattened_size, 256)  
        self.fc2 = nn.Linear(256, num_classes)

    def get_activation(self, activation):
        if activation == 'sigmoid':
            return nn.Sigmoid()
        if activation == 'relu':
            return nn.ReLU()
        if activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

    def calculate_conv_output_size(self, input_size, kernel_size, stride, padding):
        h_in, w_in = input_size
        w_out = ((w_in - kernel_size + 2 * padding) // stride) + 1
        h_out = ((h_in - kernel_size + 2 * padding) // stride) + 1
        return (h_out, w_out)

    def calculate_pool_output_size(self, input_size, kernel_size, stride):
        h_in, w_in = input_size
        w_out = ((w_in - kernel_size) // stride) + 1
        h_out = ((h_in - kernel_size) // stride) + 1
        return (h_out, w_out)

    def forward(self, x):
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        #     if i < len(self.activations):
        #         x = self.activations[i](x)

        for layer, activation in zip(self.layers, self.activations + [None] * (len(self.layers) - len(self.activations))):
          x = layer(x)
          if activation:
            x = activation(x)


        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def train_model(self, train_loader, val_loader, optimizer, loss_function, epochs=5):
     for epoch in range(1, epochs + 1):
        self.train()
        total_loss = 0.0
        print(f"Epoch {epoch}/{epochs}:")

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = self(inputs)

            labels = labels.float().view(-1, 1) if self.task == 'regression' else labels
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        self.eval()
        correct_predictions = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self(inputs)
                predictions = torch.round(outputs) if self.task == 'regression' else torch.max(outputs, 1)[1]
                labels = labels.float().view(-1, 1) if self.task == 'regression' else labels

                total_samples += labels.size(0)
                correct_predictions += (predictions == labels).sum().item()

        val_accuracy = 100 * correct_predictions / total_samples
        print(f"Training Loss: {total_loss / len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")


    def test_model(self, test_loader):
     correct_predictions = 0
     total_samples = 0
     self.eval()

     with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = self(inputs)

            if self.task == 'regression':
                predictions = torch.round(outputs)
                labels = labels.float().view(-1, 1)
            else:
                predictions = torch.argmax(outputs, dim=1)
                
            total_samples += labels.size(0)
            correct_predictions += (predictions == labels).sum().item()

     accuracy = (correct_predictions / total_samples) * 100
     print(f"Test Accuracy: {accuracy:.2f}%")



    def load(self, from_path='./best.pth'):
        self.load_state_dict(torch.load(from_path))
        print(f"Model Loaded Successfully from {from_path}")

    def save(self, to_path='./best.pth'):
        torch.save(self.state_dict(), to_path)
        print(f"Model Saved Successfully to {to_path}")
        
    