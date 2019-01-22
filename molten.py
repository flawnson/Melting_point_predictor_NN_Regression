"""
This project is meant as a precursor to project deliquesces.
The purpose of this project is to train a neural network on the patterns found in SMILES strings, and teach it the patterns in melting points
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim

"""OPENING DATASET"""
# Open dataset
data = pd.read_csv('BradleyMeltingPointDatasetClean.csv')

# Define all datasets to be used in training and turn them into lists
smiles_strings = np.asarray(list(data.smiles))
melting_points_celcius = np.asarray(list(data.mpC))

assert len(smiles_strings) == len(melting_points_celcius), "X data and Y data are not the same size"

# Define all ID points
molecule_name = list(data.name)

"""X (SMILES STRING) DATA PREPROCESSING"""
# Create unique character dictionary
charset = set("".join(list(data.smiles)))
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smiles) for smiles in data.smiles]) + 1
charset_len = len(charset)
dataset = len(smiles_strings)

print (str(charset))
print ("Number of training samples: %s" %dataset)
print ("Number of unique characters: %s" %charset_len)
print ("Maximum length of embedding after padding: %s" %embed)

# Integerize (using the character dictionary we built) and normalize (by dividing my the number of unique characters) the dataset
int_norm_smiles = []

def int_and_norm(smiles_strings):
  for smiles in smiles_strings:
    int_norm_smiles.append([(char_to_int[char] / charset_len) for char in smiles])

int_and_norm(smiles_strings)

# Pad each of the normalized and integerized smiles strings to be as long as the longest smile string in the dataset +1
pad_int_norm_smiles = []

def pad(int_norm_smiles):
  for smiles in int_norm_smiles:
    diff = embed - len(smiles)
    diff = [0] * diff
    smiles = smiles + diff
    pad_int_norm_smiles.append(smiles)
  return pad_int_norm_smiles

pad(int_norm_smiles)

"""Y (MELTING POINT TEMPERATES IN CELCIUS) DATA PREPROCESSING"""
# Normalize the melting point temperatures in a range of -1 to 1 (to account for the negative values of the dataset)
def norm(melting_points):
  norm_melting_points_celcius = 2 * (melting_points - np.min(melting_points)) / np.ptp(melting_points) - 1
  return norm_melting_points_celcius
  
norm_melting_points_celcius = norm(melting_points_celcius)

# Store in final variables and final check to ensure datasets are the same length
X = pad_int_norm_smiles
Y = norm_melting_points_celcius

assert len(pad_int_norm_smiles) == len(norm_melting_points_celcius), "X data and Y data are not the same size"

"""SPLITTING DATA PREPROCESSING"""
#Split dataset into training and testing sets
def data_splitter(dataset, train_percent):
    dataset_len = int(len(dataset))

    split_length = int(train_percent * dataset_len)
    return dataset[:split_length], dataset[split_length:]

X_data_train, X_data_test = data_splitter(pad_int_norm_smiles, .9)
Y_data_train, Y_data_test = data_splitter(norm_melting_points_celcius, .9)

# Turn datasets into tensors as input for Neural Network model
def tensorizer(datasets):
  datasets = np.asarray(datasets)
  datasets = torch.FloatTensor(datasets)
  for dataset in datasets:
    dataset = dataset.view(-1, embed)
  return datasets
    
X_data_train = tensorizer(X_data_train)
X_data_test = tensorizer(X_data_test)

print (X_data_train)
print (X_data_test)

print (Y_data_train)
print (Y_data_test)
"""NEURAL NETWORK MODEL"""
class Neural_Network(nn.Module):
        def __init__(self):
                super(Neural_Network, self).__init__()
                self.layer_1 = nn.Linear(embed, 188) # Embed is used to define numerb of input neurons (embed = length of padded smiles)
                self.layer_2 = nn.Linear(188, 94)
                self.layer_3 = nn.Linear(94, 47)
                self.layer_4 = nn.Linear(47, 13)
                self.layer_5 = nn.Linear(13, 7)
                self.layer_5 = nn.Linear(7, 1)

                self.sigmoid = nn.Sigmoid()
                self.relu = nn.ReLU()

        def forward(self, X):
                out_1 = self.relu(self.layer_1(X))
                out_2 = self.sigmoid(self.layer_2(out_1))
                out_3 = self.sigmoid(self.layer_3(out_2))
                out_4 = self.sigmoid(self.layer_4(out_3))
                prediction = self.relu(self.layer_5(out_4))
                return prediction

# Creating a variable for our model
model = Neural_Network()

"""COMPILING, FITTING AND RUNNING"""

# Construct loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Construct training loop
epochs = 1000

running_loss = 0.0
for epoch in range(epochs):

        # Define Variables
        prediction = model(X)
        loss = criterion(prediction, Y)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch % 10 == 9:
            print('Epochs: %5d | Loss: %.3f' % (epoch + 1, running_loss / 10))
            running_loss = 0.0
              
"""CHECKPOINTS"""
# # Define checkpoints (used to save the weights at each epoch, so that the model doesn't need to be retrained)
# filepath="molten-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
# callbacks_list = [checkpoint]