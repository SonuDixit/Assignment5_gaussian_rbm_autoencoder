
(b) Apply LDA on the PCA reduced images to 9 dimensions, and train a DNN with
dimensions (9, 256, 256, 10) with same output non-linearity for digit classification.
(c) Train a Gaussian Bernoulli RBM model with PCA input and 9 dimensional hidden
activation. Once the RBM is trained, forward pass the PCA input and use the 9
dimensional hidden activation from RBM for training the digit classification similar
to LDA case (using the same DNN configuration).
(d) Train an autoencoder using 25 PCA reduced image with size (25, 128, 9, 128, 25). Use
a ReLU activation and MSE loss for the Autoencoder. Following the training of the
autoencoder, generate the embeddings from 9 dimensional latent layer and use them
for training a DNN similar to the LDA case.
