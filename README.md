# Melting_point_predictor_NN_Regression

*Further details provided in the documentation within the melting.py file.*

## Project
This is a melting point temperature predictor I built in Python using the Pytorch library and NumPy as well as Pandas Python packages. All data preprocessing, model architecture, checkpoints, as well as predictions are within the melting.py file. The model receives padded, normalized, and integerized SMILES strings as input to output a predicted melting point. The dataset consists of a couple thousand labeled datapoints in a csv file format.

## Article
Here's a brief article summarizing my inspiration, intention, and execution, as well as a look into potential applications.
https://towardsdatascience.com/the-great-molasses-flood-predicting-the-melting-point-of-metals-1c4440d2edd2

## Future plans and outcomes
The ultimate goal of this project wis to train a neural network model on the inherent properties of a SMILES string such that it can learn the correlation between the string and its corresponding melting point temperature. Project melting is a precursor to Project Deliquesces, where, combined with concepts and methods used in Project Novel (a molecule generating RNN built with Keras), the goal will be to generate new molecules with a specified and/or desired melting point temperature. Upwards and onwards, always and only :rocket:!
