# DL from scratch
Build wheels for deep learning model Irregularly.
- - -
## 1. MLP in MNIST (Python+Numpy version)
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
