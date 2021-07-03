#include "../include/utils.hpp"
using namespace std;

void test01()
{
    // 测试标签
    Matrix<unsigned> labels(60000,1);
    load_mnist_label("../data/MNIST/train-labels-idx1-ubyte",labels);
    for (size_t i = 0; i < 10; i++)
    {
        cout << "label " << i << ": " << labels.ma[i][0] << endl;
    }
    // 测试图片
    Matrix<_Float32> images(60000,784);
    load_mnist_images("../data/MNIST/train-images-idx3-ubyte",images);
    cout << "images 0: " << endl;
    for (size_t i = 0; i < 28; i++)
    {
        for (size_t j = 0; j < 28; j++)
        {
            cout << images.ma[0][i*28 + j] << " ";
        }
        cout << endl;
    }
    
    
}

int main()
{
    test01();
}