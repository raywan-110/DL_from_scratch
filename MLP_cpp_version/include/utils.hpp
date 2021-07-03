#pragma once
#include "Matrix.hpp"
#include <fstream>
using namespace std;
//g++ 不支持模板的分离编译，需要把函数实现写在头文件里面

/**
 * @brief 
 * 1. 激活函数及其导数
 */
template <class dataType>
Matrix<dataType> relu(Matrix<dataType> z)
{
    Matrix<dataType> res(z.row, z.col);
    for (size_t i = 0; i < z.row; i++)
    {
        for (size_t j = 0; j < z.col; j++)
        {
            if (z.ma[i][j] < 0.)
            {
                res.ma[i][j] = 0.;
            }
            else
            {
                res.ma[i][j] = z.ma[i][j];
            }
        }
    }
    return res;
}

template <class dataType>
Matrix<dataType> relu_D(Matrix<dataType> z)
{
    Matrix<dataType> res(z.row, z.col);
    for (size_t i = 0; i < z.row; i++)
    {
        for (size_t j = 0; j < z.col; j++)
        {
            if (z.ma[i][j] < 0.)
            {
                res.ma[i][j] = 0.;
            }
            else
            {
                res.ma[i][j] = 1.;
            }
        }
    }
    return res;   
}

/**
 * @brief 
 * 2. 交叉熵损失函数及其导数(处理上下溢出)
 */
template <class dataType>
Matrix<dataType> cross_entrpy(Matrix<dataType> z_last, Matrix<unsigned> label)
{
    Matrix<dataType> z_exp = Matrix<dataType>::Mexp(z_last - z_last.valmax(0)); // 按行取最大值后，减去每行的最大值
    Matrix<dataType> partition = z_exp.sum(0); // 按行求和
    Matrix<dataType> activation = z_exp / partition;

    // 根据label取出对应的probs
    Matrix<dataType> probs(activation.row,1); // row是batch size
    for (size_t i = 0; i < activation.row; i++)
    {
        probs.ma[i][0] = activation.ma[i][label.ma[i][0]]; // label的数据类型是int
    }
    Matrix<dataType> res = Matrix<dataType>::Mlog(probs + 5e-5);
    return res.sum().inverse() / z_last.row;
}

template<class dataType>
Matrix<dataType> cross_entrpy_D(Matrix<dataType> z_last, Matrix<unsigned> label)
{
    Matrix<dataType> z_exp = Matrix<dataType>::Mexp(z_last - z_last.valmax(0));
    Matrix<dataType> partition = z_exp.sum(0);
    Matrix<dataType> activation = z_exp / partition;
    for (size_t i = 0; i < activation.row; i++)
    {
        activation.ma[i][label.ma[i][0]] -= 1.; // 减去one-hot code
    }
    return activation;
}

/**
 * @brief parse and load mnist dataset
 * 
 */

int swap_endian(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255; // 高8位
    ch2 = (i >> 8) & 255; // 高8位
    ch3 = (i >> 16) & 255; // 高8位
    ch4 = (i >> 24) & 255; // 高8位
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

template<class dataType>
void load_mnist_label(string filename, Matrix<dataType> &labels)
{
    ifstream file(filename,ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number,sizeof(magic_number)); // 读入4bytes魔数
        magic_number = swap_endian(magic_number); // 交换字节序为小字节序
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = swap_endian(number_of_labels);
        cout << "magic number = " << magic_number << endl;
        cout << "number of labels = " << number_of_labels << endl;
        for (size_t i = 0; i < number_of_labels; i++)
        {
            unsigned char label = 0; // 8 bits
            file.read((char*)&label,sizeof(label));
            labels.ma[i][0] = (dataType) label;
        }
        file.close();
    }
    else
    {
        throw "no this file!";
    }
}

template<class dataType>
void load_mnist_images(string filename, Matrix<dataType> &images)
{
    ifstream file(filename,ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number,sizeof(magic_number)); // 读入4bytes魔数
        magic_number = swap_endian(magic_number); // 交换字节序为小字节序
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = swap_endian(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows = swap_endian(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = swap_endian(n_cols);
        cout << "magic number = " << magic_number << endl;
        cout << "number of images = " << number_of_images << endl;
        cout << "rows = " << n_rows << endl;
        cout << "cols = " << n_cols << endl;
        for (size_t i = 0; i < number_of_images; i++)
        {
            for (size_t j = 0; j < n_rows*n_cols; j++)
            {
                unsigned char image = 0;
                file.read((char *)&image,sizeof(image)); // 读入一个像素
                images.ma[i][j] = (dataType)image;
            }
        }
    }
    
}


