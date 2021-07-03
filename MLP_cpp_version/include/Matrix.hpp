#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <assert.h>
using namespace std;

/*实现矩阵类，作为网络运算的底层结构*/
template<class dataType = _Float32>
class Matrix
{
public:
    /*采用vector<vector<datatype>>存储数据, 构造之前并不知道矩阵大小*/
    vector<vector<dataType>> ma;
    int col, row;
public:
    /*构造与析构方法*/
    Matrix(int row = 0, int col = 0); // 默认构造全0的矩阵
    Matrix(const Matrix &m); // 复制构造函数
    Matrix(dataType * m, int row, int col); // 从普通数组中构造matrix
    ~Matrix();

    /*信息获取方法*/
    static void printMatrix(const Matrix &m);
    pair<int, int> shape();

    /**
     * @brief 
     * 矩阵运算
     * 1. 转置(transpose)
     * 2. 矩阵乘法
     * 3. 矩阵中的最大/小值
     * @return Matrix 
     */
    Matrix transpose(); // 转置
    Matrix dot(const Matrix & m); // 矩阵乘法
    Matrix valmax(int dim = -1); // 最大值
    Matrix valmin(int dim = -1); // 最小值
    Matrix<unsigned> argmax(int dim = 0); // 计算最大值的坐标
    Matrix<unsigned> argmin(int dim = 0); // 计算最小值的坐标
    Matrix sum(int dim = -1); // 求和

    /**
     * @brief 
     * 逐元素运算 (广播机制)
     * 1. 按元素加法
     * 2. 按元素减法
     * 3. 按元素乘法
     * 4. 按元素除法
     * @param m  class Matrix 
     * @param x dataType value
     * @return Matrix 
     */
    Matrix operator+(const Matrix & m);
    Matrix operator+(const dataType x);
    Matrix operator-(const Matrix & m);
    Matrix operator-(const dataType x);
    Matrix operator*(const Matrix & m);
    Matrix operator*(const dataType x);
    Matrix operator/(const Matrix & m);
    Matrix operator/(const dataType x);
    Matrix inverse();

    void normalize(); // 归一化矩阵

    static Matrix Mexp(const Matrix & m);
    static Matrix Mlog(const Matrix & m);
    static Matrix Msqrt(const Matrix & m);




};


/*构造与析构方法*/
template<class dataType>
Matrix<dataType>::Matrix(int row, int col)
{
    this->row = row;
    this->col = col;
    this->ma.resize(row);
    for (size_t i = 0; i < ma.size(); i++)
    {
        ma[i].resize(col);
        for (size_t j = 0; j < ma[i].size(); j++)
        {
            ma[i][j] = 0.;
        }
    }
}

template<class dataType>
Matrix<dataType>::Matrix(const Matrix & m)
{
    this->ma = m.ma;
    this->row = m.row;
    this->col = m.col;
}

template<class dataType>
Matrix<dataType>::Matrix(dataType * m, int row, int col)
{
    this->row = row;
    this->col = col;
    this->ma.resize(row);
    for (size_t i = 0; i < row; i++)
    {
        this->ma[i].resize(col);
        for (size_t j = 0; j < col; j++)
        {
            this->ma[i][j] = m[i*col + j]; // 手动计算出每一行的偏移
        }
    }
}

template<class dataType>
Matrix<dataType>::~Matrix()
{
    this->ma.resize(0);
    vector<vector<dataType>> (this->ma).swap(this->ma); // 利用swap释放空间
}

/*信息获取方法*/
template<class dataType>
void Matrix<dataType>::printMatrix(const Matrix &m)
{
    for (size_t i = 0; i < m.ma.size(); i++)
    {
        for (size_t j = 0; j < m.ma[i].size(); j++)
        {
            cout << m.ma[i][j] << " ";
        }
        cout << endl;
    }
}

template<class dataType>
pair<int,int> Matrix<dataType>::shape()
{
    return make_pair(this->row, this->col);
}


/*矩阵运算*/
template<class dataType>
Matrix<dataType> Matrix<dataType>::transpose()
{
    Matrix res(this->col,this->row);
    for (size_t i = 0; i < this->col; i++)
    {
        for (size_t j = 0; j < this->row; j++)
        {
            res.ma[i][j] = this->ma[j][i];
        }
    }
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::dot(const Matrix & m)
{
    assert(this->col == m.row);
    Matrix<dataType> res(this->row, m.col);
    for (size_t i = 0; i < res.row; i++)
    {
        for (size_t j = 0; j < res.col; j++)
        {
            dataType sum = 0;
            for (size_t k = 0; k < this->col; k++)
            {
                sum += this->ma[i][k] * m.ma[k][j];
            }
            
            res.ma[i][j] = sum;
        }
    }
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::valmax(int dim)
{   
    assert(dim == -1 || dim == 0 || dim == 1);
    if (dim == -1)
    {
        /*返回全局最大值*/
        Matrix<dataType> res(1,1);
        dataType max_value = this->ma[0][0];
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                if (this->ma[i][j] > max_value)
                {
                    max_value = this->ma[i][j];
                }
            }
        }
        res.ma[0][0] = max_value;
        return res;
    }
    else if (dim == 0)
    {
        /*返回每行的最大值*/
        Matrix<dataType> res(this->row,1);
        for (size_t i = 0; i < this->row; i++)
        {
            dataType max_value = this->ma[i][0];
            for (size_t j = 0; j < this->col; j++)
            {
                if (this->ma[i][j] > max_value)
                {
                    max_value = this->ma[i][j];
                }
            }
            res.ma[i][0] = max_value;;
        }
        return res;
    }
    else if (dim == 1)
    {
        /*返回每列的最大值*/
        Matrix<dataType> res(1,this->col);
        for (size_t i = 0; i < this->col; i++)
        {
            dataType max_value = this->ma[0][i];
            for (size_t j = 0; j < this->row; j++)
            {
                if (this->ma[j][i] > max_value)
                {
                    max_value = this->ma[j][i];
                }
            }
            res.ma[0][i] = max_value;
        }
        return res;
    }
    throw "no this option!";
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::valmin(int dim)
{   
    assert(dim == -1 || dim == 0 || dim == 1);
    if (dim == -1)
    {
        /*返回全局最大值*/
        Matrix<dataType> res(1,1);
        dataType min_value = this->ma[0][0];
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                if (this->ma[i][j] < min_value)
                {
                    min_value = this->ma[i][j];
                }
            }
        }
        res.ma[0][0] = min_value;
        return res;
    }
    else if (dim == 0)
    {
        /*返回每行的最大值*/
        Matrix<dataType> res(this->row,1);
        for (size_t i = 0; i < this->row; i++)
        {
            dataType min_value = this->ma[i][0];
            for (size_t j = 0; j < this->col; j++)
            {
                if (this->ma[i][j] < min_value)
                {
                    min_value = this->ma[i][j];
                }
            }
            res.ma[i][0] = min_value;;
        }
        return res;
    }
    else if (dim == 1)
    {
        /*返回每列的最大值*/
        Matrix<dataType> res(1,this->col);
        for (size_t i = 0; i < this->col; i++)
        {
            dataType min_value = this->ma[0][i];
            for (size_t j = 0; j < this->row; j++)
            {
                if (this->ma[j][i] < min_value)
                {
                    min_value = this->ma[j][i];
                }
            }
            res.ma[0][i] = min_value;
        }
        return res;
    }

    throw "no this option!";
}

template<class dataType>
Matrix<unsigned> Matrix<dataType>::argmax(int dim)
{   
    assert(dim == 0 || dim == 1);
    if (dim == 0)
    {
        /*返回每行的最大值的下标*/
        Matrix<unsigned> res(this->row,1);
        for (size_t i = 0; i < this->row; i++)
        {
            dataType max_value = this->ma[i][0];
            unsigned index = 0;
            for (size_t j = 0; j < this->col; j++)
            {
                if (this->ma[i][j] > max_value)
                {
                    max_value = this->ma[i][j];
                    index = j;
                }
            }
            res.ma[i][0] = index;
        }
        return res;
    }
    else if (dim == 1)
    {
        /*返回每列的最大值*/
        Matrix<unsigned> res(1,this->col);
        for (size_t i = 0; i < this->col; i++)
        {
            dataType max_value = this->ma[0][i];
            unsigned index = 0;
            for (size_t j = 0; j < this->row; j++)
            {
                if (this->ma[j][i] > max_value)
                {
                    max_value = this->ma[j][i];
                    index = j;
                }
            }
            res.ma[0][i] = index;
        }
        return res;
    }
    throw "no this option!";
}

template<class dataType>
Matrix<unsigned> Matrix<dataType>::argmin(int dim)
{   
    assert(dim == 0 || dim == 1);
    if (dim == 0)
    {
        /*返回每行的最大值的下标*/
        Matrix<unsigned> res(this->row,1);
        for (size_t i = 0; i < this->row; i++)
        {
            dataType min_value = this->ma[i][0];
            unsigned index = 0;
            for (size_t j = 0; j < this->col; j++)
            {
                if (this->ma[i][j] < min_value)
                {
                    min_value = this->ma[i][j];
                    index = j;
                }
            }
            res.ma[i][0] = index;
        }
        return res;
    }
    else if (dim == 1)
    {
        /*返回每列的最大值*/
        Matrix<unsigned> res(1,this->col);
        for (size_t i = 0; i < this->col; i++)
        {
            dataType min_value = this->ma[0][i];
            unsigned index = 0;
            for (size_t j = 0; j < this->row; j++)
            {
                if (this->ma[j][i] < min_value)
                {
                    min_value = this->ma[j][i];
                    index = j;
                }
            }
            res.ma[0][i] = index;
        }
        return res;
    }
    throw "no this option!";
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::sum(int dim)
{   
    assert(dim == -1 || dim == 0 || dim == 1);
    if (dim == -1)
    {
        /*返回全局之和*/
        Matrix<dataType> res(1,1);
        dataType sum_value = 0.;
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                sum_value += this->ma[i][j];
            }
        }
        res.ma[0][0] = sum_value;
        return res;
    }
    else if (dim == 0)
    {
        /*返回每行的和*/
        Matrix<dataType> res(this->row,1);
        for (size_t i = 0; i < this->row; i++)
        {
            dataType sum_value = 0.;
            for (size_t j = 0; j < this->col; j++)
            {
                sum_value += this->ma[i][j];
            }
            res.ma[i][0] = sum_value;
        }
        return res;
    }
    else if (dim == 1)
    {
        /*返回每列的和*/
        Matrix<dataType> res(1,this->col);
        for (size_t j = 0; j < this->col; j++)
        {
            dataType sum_value = 0.;
            for (size_t i = 0; i < this->row; i++)
            {
                sum_value += this->ma[i][j];
            }
            res.ma[0][j] = sum_value;
        }
        return res;
    }
    throw "no this option!";
}






template<class dataType>
Matrix<dataType> Matrix<dataType>::operator+(const Matrix & m)
{
    Matrix res(this->row, this->col);
    // broadcast
    if (m.col == 1)
    {
        assert(this->row == m.row);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] + m.ma[i][0];
            }
        }
    }
    else if (m.row == 1)
    {
        assert(this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] + m.ma[0][j];
            }
        }
    }
    else
    {
        assert(this->row == m.row && this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] + m.ma[i][j];
            }
        }
    }
    
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::operator+(const dataType x)
{
    Matrix res(this->row, this->col);
    for (size_t i = 0; i < this->row; i++)
    {
        for (size_t j = 0; j < this->col; j++)
        {
            res.ma[i][j] = this->ma[i][j] + x;
        }
    }

    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::operator-(const Matrix & m)
{
    Matrix res(this->row, this->col);
    // broadcast
    if (m.col == 1)
    {
        assert(this->row == m.row);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] - m.ma[i][0];
            }
        }
    }
    else if (m.row == 1)
    {
        assert(this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] - m.ma[0][j];
            }
        }
    }
    else
    {
        assert(this->row == m.row && this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] - m.ma[i][j];
            }
        }
    }
    
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::operator-(const dataType x)
{
    Matrix res(this->row, this->col);
    for (size_t i = 0; i < this->row; i++)
    {
        for (size_t j = 0; j < this->col; j++)
        {
            res.ma[i][j] = this->ma[i][j] - x;
        }
    }

    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::operator*(const Matrix & m)
{
    Matrix res(this->row, this->col);
    // broadcast
    if (m.col == 1)
    {
        assert(this->row == m.row);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] * m.ma[i][0];
            }
        }
    }
    else if (m.row == 1)
    {
        assert(this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] * m.ma[0][j];
            }
        }
    }
    else
    {
        assert(this->row == m.row && this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                res.ma[i][j] = this->ma[i][j] * m.ma[i][j];
            }
        }
    }
    
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::operator*(const dataType x)
{
    Matrix res(this->row, this->col);
    for (size_t i = 0; i < this->row; i++)
    {
        for (size_t j = 0; j < this->col; j++)
        {
            res.ma[i][j] = this->ma[i][j] * x;
        }
    }

    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::operator/(const Matrix & m)
{
    Matrix res(this->row, this->col);
    // broadcast
    if (m.col == 1)
    {
        assert(this->row == m.row);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                assert(m.ma[i][0] != 0);
                res.ma[i][j] = this->ma[i][j] / m.ma[i][0];
            }
        }
    }
    else if (m.row == 1)
    {
        assert(this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                assert(m.ma[0][i] != 0);
                res.ma[i][j] = this->ma[i][j] / m.ma[0][j];
            }
        }
    }
    else
    {
        assert(this->row == m.row && this->col == m.col);
        for (size_t i = 0; i < this->row; i++)
        {
            for (size_t j = 0; j < this->col; j++)
            {
                assert(m.ma[i][j] != 0);
                res.ma[i][j] = this->ma[i][j] / m.ma[i][j];
            }
        }
    }
    
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::operator/(const dataType x)
{
    assert(x != 0);
    Matrix res(this->row, this->col);
    for (size_t i = 0; i < this->row; i++)
    {
        for (size_t j = 0; j < this->col; j++)
        {
            res.ma[i][j] = this->ma[i][j] / x;
        }
    }

    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::inverse()
{
    Matrix<dataType> res(this->row,this->col);
    for (size_t i = 0; i < this->row; i++)
    {
        for (size_t j = 0; j < this->col; j++)
        {
            res.ma[i][j] = -this->ma[i][j];
        }
    }
    return res;
    
}

template<class dataType>
void Matrix<dataType>::normalize()
{
    // 计算均值
    dataType mean = this->sum(-1).ma[0][0] / (this->row * this->col);
    // 计算标准差
    dataType std = 0.;
    for (size_t i = 0; i < this->row; i++)
    {
        for (size_t j = 0; j < this->col; j++)
        {
            std += (this->ma[i][j] - mean) * (this->ma[i][j] - mean);
        }
    }
    std = sqrt(std / (this->row * this->col));
    // 逐元素归一化
    for (size_t i = 0; i < this->row; i++)
    {
        for (size_t j = 0; j < this->col; j++)
        {
            this->ma[i][j] = (this->ma[i][j] - mean) / std;
        }
    }
}
template<class dataType>
Matrix<dataType> Matrix<dataType>::Mexp(const Matrix & m)
{
    Matrix res(m.row,m.col);
    for (size_t i = 0; i < m.row; i++)
    {
        for (size_t j = 0; j < m.col; j++)
        {
            res.ma[i][j] = exp(m.ma[i][j]);
        }
        
    }
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::Mlog(const Matrix & m)
{
    Matrix res(m.row,m.col);
    for (size_t i = 0; i < m.row; i++)
    {
        for (size_t j = 0; j < m.col; j++)
        {
            res.ma[i][j] = log(m.ma[i][j]);
        }
        
    }
    return res;
}

template<class dataType>
Matrix<dataType> Matrix<dataType>::Msqrt(const Matrix & m)
{
    Matrix res(m.row,m.col);
    for (size_t i = 0; i < m.row; i++)
    {
        for (size_t j = 0; j < m.col; j++)
        {
            res.ma[i][j] = sqrt(m.ma[i][j]);
        }
        
    }
    return res;
    
}


