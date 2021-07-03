#pragma once
#include "Matrix.hpp"
#include "utils.hpp"
#include <random>
#include <chrono>
#include <algorithm>
using namespace std;

template<class dataType = _Float32>
class net
{
public:
    vector<Matrix<dataType>> weights; // 权重
    vector<Matrix<dataType>> bias; // 偏差
    vector<Matrix<dataType>> activations; // 保留前向传播中的激活值
    vector<Matrix<dataType>> p_output; // 保留前向传播中的净输出
    int num_layers; // 网路层数
    bool train = true;

public:
    /**
     * @brief 构造与析构函数
     * 
     * @param layers 
     */
    net(vector<int> layers);
    ~net();

    /**
     * @brief 切换训练状态
     * 
     */
    void set_train();
    void set_val();

    /**
     * @brief 前向传播与反向传播
     * 
     * @param a 
     * @return Matrix<dataType> 
     */
    Matrix<dataType> fp(Matrix<dataType> a);
    pair<vector<Matrix<dataType>>, vector<Matrix<dataType>>> bp(Matrix<dataType> z_last, Matrix<unsigned> label); // 返回weights与bias的梯度
    
    /**
     * @brief 参数更新
     * 
     * @param grad 
     */
    void step(pair<vector<Matrix<dataType>>,vector<Matrix<dataType>>> grad, _Float32 lr , _Float32 weights_decay=0.);


};

template<class dataType>
net<dataType>::net(vector<int> layers)
{
    /*e.g. size = [3,4,2]*/
    this->num_layers = layers.size();
    // 随机初始化权重与偏差
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed); // 随机数生成引擎
    normal_distribution<_Float32> dist(0.,1.);
    for (size_t i = 0; i < this->num_layers - 1; i++)
    {
        // weights
        Matrix<dataType> w(layers[i],layers[i+1]);
        for (size_t j = 0; j < w.row; j++)
        {
            for (size_t k = 0; k < w.col; k++)
            {
                w.ma[j][k] = dist(generator) / sqrt(layers[i+1]);
            }
        }
        // bias初始化为0
        Matrix<dataType> b(1,layers[i+1]);
        this->weights.push_back(w);
        this->bias.push_back(b);
    }
    
}

template<class dataType>
net<dataType>::~net()
{
    // 清空所有的容器
    this->weights.resize(0);
    vector<Matrix<dataType>> (this->weights).swap(this->weights);
    this->bias.resize(0);
    vector<Matrix<dataType>> (this->bias).swap(this->bias);
    this->activations.resize(0);
    vector<Matrix<dataType>> (this->activations).swap(this->activations);
    this->p_output.resize(0);
    vector<Matrix<dataType>> (this->p_output).swap(this->p_output);
}

template<class dataType>
void net<dataType>::set_train()
{
    this->train = true;
}

template<class dataType>
void net<dataType>::set_val()
{
    this->train = false;
}

template<class dataType>
Matrix<dataType> net<dataType>::fp(Matrix<dataType> a)
{
    /*input: batch data -> batch_size*input_dim 
    output:  z_last -> batch_size*output_dim*/
    
    if (this->train)
    {
        this->activations.clear();
        this->p_output.clear();
        this->activations.push_back(a); // 第一个激活值
    }
    Matrix<dataType> act = a;
    for (size_t i = 0; i < this->num_layers - 2; i++)
    {
        Matrix<dataType> z = act.dot(this->weights[i]) + this->bias[i];
        act = relu<dataType>(z); 
        if (this->train)
        {
            this->activations.push_back(act);
            this->p_output.push_back(z);
        }
    }

    Matrix<dataType> z_last = act.dot(this->weights.back()) + this->bias.back();
    return z_last;
}
template<class dataType>
pair<vector<Matrix<dataType>>, vector<Matrix<dataType>>> net<dataType>::bp(Matrix<dataType> z_last, Matrix<unsigned> label)
{
    // 保存梯度的容器
    vector<Matrix<dataType>> grad_b;
    vector<Matrix<dataType>> grad_w;
    // 计算损失函数对最后一层净输出的导数(误差)
    Matrix<dataType> delta = cross_entrpy_D(z_last,label);
    // 更新最后一层的权重
    grad_b.push_back(delta.sum(1) / z_last.row);
    grad_w.push_back(this->activations.back().transpose().dot(delta) / z_last.row);
    this->activations.pop_back();
    // 开始反向传播
    for (size_t i = 0; i < this->num_layers-2; i++)
    {
        Matrix<dataType> grad_a2z = relu_D(this->p_output.back());
        this->p_output.pop_back();
        delta = grad_a2z * (delta.dot(this->weights[num_layers-2-i].transpose()));
        grad_b.push_back(delta.sum(1) / z_last.row);
        grad_w.push_back(this->activations.back().transpose().dot(delta) / z_last.row);
        this->activations.pop_back();
    }
    // 反转与网络顺序对应
    reverse(grad_b.begin(),grad_b.end());
    reverse(grad_w.begin(),grad_w.end());
    return make_pair(grad_b,grad_w);
}

template<class dataType>
void net<dataType>::step(pair<vector<Matrix<dataType>>,vector<Matrix<dataType>>> grad, _Float32 lr , _Float32 weights_decay)
{
    // 更新 bias
    for (size_t i = 0; i < grad.first.size(); i++)
    {
        this->bias[i] = this->bias[i] - grad.first[i] * lr;
        // Matrix<dataType>::printMatrix(grad.first[i]);
    }
    // 更新 weights
    for (size_t i = 0; i < grad.second.size(); i++)
    {
        if (weights_decay != 0.)
        {
            this->weights[i] = this->weights[i] - (this->weights[i] * weights_decay + grad.second[i]) * lr;
        }
        else
        {
            this->weights[i] = this->weights[i] - grad.second[i] * lr;
        }
    }
}
