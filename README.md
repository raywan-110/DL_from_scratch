# DL from scratch
Build wheels for deep learning model Irregularly.
- - -
## 1. MLP on MNIST (Python+Numpy version)
### Usage
  ```python
  python ./mnist_from_scratch.py
  ```
  BTW, you need to download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ and unzip it by yourself. **Don't forget to set the data set path correctly**.
### Results
  - **Hyperparameters setting**
  **epochs**: 100  
  **batch_size**: 128  
  **lr**: 0.01
  - **model structure**  
  **fully connected layers**: [784 512 10]  

  | Activation function | Regularization| accuracy in test set|
  |:--: | :--: | :--: |
  | sigmoid|L2 | 96.87% |
  | relu |None | 97.97% |  
  | relu |L2 | 97.98% |  

### Summary
- mind the **numerical overflow problems** brought by the exp(·) operation(**convert to  a mathematically equivalent form to overcome them**);
- convert the labels into **one-hot format** when computing gradient of cross-entropy loss(**incorrect calculation of gradients leads to non-convergence of training**);
- Pay attention to the **byte order**(little/big endian) of storage when parsing the data set.
