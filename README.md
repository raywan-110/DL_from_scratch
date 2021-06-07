# DL from scratch
Build wheels for deep learning model Irregularly.
- - -
## 1. MLP in MNIST (Python+Numpy version)
### Basic Principle for Backward Propagation in MLP
$$
z^{l} = a^{l-1}W+b\ \ \ (a^{0}=X)\\
a^{l} = \sigma(z^l)\ \ \ (z^{last}=\hat{y})\\
\frac{\partial{L(y,\hat{y})}}{\partial{W^{l}}}=(a^{l-1})^{T}\delta^{l}\ \ \in(M_{l-1}\times M_l) \\
\frac{\partial{L(y,\hat{y})}}{\partial{b^{l}}}=\delta^{l}\ \ \in(1\times M_l)\\
\delta^{l}=\sigma(z^{l})'\odot[\delta^{l+1}(W^{l+1})^T]\ \ \in(1\times M_l)
$$
### Usage
  ```python
  python ./mnist_from_scratch.py
  ```
  BTW, you need to download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and unzip it by yourself. Don't forget to set the data set path correctly.
### Results
  - **Hyperparameters setting**
  **epochs**: 100  
  **batch_size**: 128  
  **lr**: 0.01
  - **model structure**  
  **fully connected layers**: $784\times 512\times 10$  

  | Activation function | Regularization| accuracy in test set|
  |-- | -- | -- |
  | sigmoid|L2 | |
  | tanh |L2 | |  
  | relu |L2 | |  
