#include "../include/Matrix.hpp"
#include <random>
#include <algorithm>
#include <chrono>
using namespace std;

/**
 * @brief 
 * 测试构造函数 与基本信息显示方法
 */
void test01()
{
    // 1. 传入行列的构造函数
    Matrix<_Float32> ma1 = Matrix<_Float32>(4,3);
    cout << "ma1:" << endl;
    Matrix<_Float32>::printMatrix(ma1);
    // 2. 复制构造函数
    Matrix<_Float32> ma2(ma1);
    cout << "ma2:" << endl;
    Matrix<_Float32>::printMatrix(ma2.transpose());
    // 3. 从普通数组构造
    _Float32 array[4][3] = {0}; // 可以理解为 _Float32 * [3], array关联的试是_Float32 * [3]类型，因此不能用关联Float32 * 类型的二维指针直接指向
    _Float32 k = 0.;
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            array[i][j] = k;
            k += 1.1;
        }
    }
    cout <<"float32 array: " << endl;
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            cout << array[i][j] << " ";
        }
        cout << endl;
    }
    cout << "ma:" << endl;
    Matrix<_Float32> ma3((_Float32 *)array,4,3);
    Matrix<_Float32>::printMatrix(ma3);
    pair<int,int> shape = ma3.shape();
    cout << "shape of ma3:" << endl;
    cout << "row: " << shape.first << endl;
    cout << "col: " << shape.second << endl;
}

/**
 * @brief 
 * 测试矩阵运算方法
 */
void test02()
{
    // 1. 矩阵乘法与转置
    Matrix<_Float32> ma1(2,3);
    float k = 0;
    for (size_t i = 0; i < ma1.row; i++)
    {
        for (size_t j = 0; j < ma1.col; j++)
        {
            ma1.ma[i][j] = k;
            k += 1;
        }
    }
    cout << "ma1:" << endl;
    Matrix<_Float32>::printMatrix(ma1);
    Matrix<_Float32> ma2(3,3);
    for (size_t i = 0; i < ma2.row; i++)
    {
        ma2.ma[i][i] = 1;
    }
    cout << "ma2:" << endl;
    Matrix<_Float32>::printMatrix(ma2);
    Matrix<_Float32> ma3 = ma1.dot(ma2);
    cout << "ma1.dot(ma2):" << endl;
    Matrix<_Float32>::printMatrix(ma3);
    cout << "ma1.dot(ma1.transpose()):" << endl;
    Matrix<_Float32>::printMatrix(ma3.dot(ma3.transpose()));
    Matrix<_Float32> ma4(3,1);
    for (size_t i = 0; i < ma4.row; i++)
    {
        ma4.ma[i][0] = 1;
    }
    cout << "ma4:" << endl;
    Matrix<_Float32>::printMatrix(ma4);
    cout << "ma1.dot([1,1,1]]):" << endl;
    Matrix<_Float32>::printMatrix(ma3.dot(ma4));
    // 2. 最大值与最小值
    cout << "ma1 max value in row:" << endl;
    Matrix<_Float32>::printMatrix(ma1.valmax(0));
    cout << "ma1 max value in col:" << endl;
    Matrix<_Float32>::printMatrix(ma1.valmax(1));
    cout << "ma1 max value:" << endl;
    Matrix<_Float32>::printMatrix(ma1.valmax());

    cout << "ma1 min value in row:" << endl;
    Matrix<_Float32>::printMatrix(ma1.valmin(0));
    cout << "ma1 min value in col:" << endl;
    Matrix<_Float32>::printMatrix(ma1.valmin(1));
    cout << "ma1 min value:" << endl;
    Matrix<_Float32>::printMatrix(ma1.valmin());
    cout << "ma1 sum value in row:" << endl;
    Matrix<_Float32>::printMatrix(ma1.sum(0));
    cout << "ma1 sum value in col:" << endl;
    Matrix<_Float32>::printMatrix(ma1.sum(1));
    cout << "ma1 sum value:" << endl;
    Matrix<_Float32>::printMatrix(ma1.sum());
    cout << "ma1 argmax in row:" << endl;
    Matrix<unsigned>::printMatrix(ma1.argmax(0));
    cout << "ma1 argmax in col:" << endl;
    Matrix<unsigned>::printMatrix(ma1.argmax(1));
    cout << "ma1 argmin in row:" << endl;
    Matrix<unsigned>::printMatrix(ma1.argmin(0));
    cout << "ma1 argmin in col:" << endl;
    Matrix<unsigned>::printMatrix(ma1.argmin(1));
}

/**
 * @brief 
 * 测试按元素运算
 */
void test03()
{
    // 1. 矩阵乘法与转置
    Matrix<_Float32> ma1(2,3);
    float k = 0;
    for (size_t i = 0; i < ma1.row; i++)
    {
        for (size_t j = 0; j < ma1.col; j++)
        {
            ma1.ma[i][j] = k;
            k += 1;
        }
    }
    cout << "ma1:" << endl;
    Matrix<_Float32>::printMatrix(ma1);
    
    Matrix<_Float32> ma2(2,3);
    for (size_t i = 0; i < ma1.row; i++)
    {
        for (size_t j = 0; j < ma1.col; j++)
        {
            ma2.ma[i][j] = k;
            k += 1;
        }
    }
    cout << "ma2:" << endl;
    Matrix<_Float32>::printMatrix(ma2);
    // 用于broadcast测试
    Matrix<_Float32> ma3(2,1);
    Matrix<_Float32> ma4(1,3);
    for (size_t i = 0; i < ma3.row; i++)
    {
        ma3.ma[i][0] = 2;
    }
    for (size_t i = 0; i < ma4.col; i++)
    {
        ma4.ma[0][i] = 3;
    }
    cout << "ma3:" << endl;
    Matrix<_Float32>::printMatrix(ma3);
    cout << "ma4:" << endl;
    Matrix<_Float32>::printMatrix(ma4);

    // 测试 + 
    cout << "ma1 + ma2 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 + ma2);
    cout << "ma1 + ma3 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 + ma3);
    cout << "ma1 + ma4 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 + ma4);
    cout << "ma1 + 3.14 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 + 3.14);
    // 测试 - 
    cout << "ma1 - ma2 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 - ma2);
    cout << "ma1 - ma3 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 - ma3);
    cout << "ma1 - ma4 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 - ma4);
    cout << "ma1 - 3.14 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 - 3.14);
    // 测试 * 
    cout << "ma1 * ma2 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 * ma2);
    cout << "ma1 * ma3 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 * ma3);
    cout << "ma1 * ma4 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 * ma4);
    cout << "ma1 * 3.14 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 * 3.14);
    // 测试 /
    cout << "ma1 / ma2 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 / ma2);
    cout << "ma1 / ma3 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 / ma3);
    cout << "ma1 / ma4 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 / ma4);
    cout << "ma1 / 3.14 = " << endl;
    Matrix<_Float32>::printMatrix(ma1 / 3.14);
    // 测试 Mexp Mlog Msqrt
    cout << "exp(ma1) = " << endl;
    Matrix<_Float32>::printMatrix(Matrix<_Float32>::Mexp(ma1));
    cout << "log(ma1) = " << endl;
    Matrix<_Float32>::printMatrix(Matrix<_Float32>::Mlog(ma1));
    cout << "sqrt(ma1) = " << endl;
    Matrix<_Float32>::printMatrix(Matrix<_Float32>::Msqrt(ma1));
    cout << "normalize(ma1) = " << endl;
    ma1.normalize();
    Matrix<_Float32>::printMatrix(ma1);
}

void test04()
{
    Matrix<_Float32> m(3,4);
    // 生成高斯矩阵
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator(seed); // 随机数生成引擎
    normal_distribution<_Float32> dist(0,1);
    for (size_t i = 0; i < 10; i++)
    {
        cout << "generate: " << dist(generator) << endl;
    }
    vector<int> permutation;
    for (size_t i = 0; i < 10; i++)
    {
        permutation.push_back(i);
    }
    cout << "iterator plus number:" << endl;
    vector<int> tmp;
    tmp.assign(permutation.begin(),permutation.begin() + 5);
    cout << "tmp: " << endl;
    for (size_t i = 0; i < tmp.size(); i++)
    {
        cout << tmp[i] << " ";
    }
    cout << endl;
    for (size_t i = 0; i < 3; i++)
    {
        srand(seed);
        random_shuffle(permutation.begin(),permutation.end());
        for (size_t i = 0; i < 10; i++)
        {
            cout << permutation[i] << " " << endl;
        }
        cout << endl;
    }
    
    
}
int main()
{
    // test01();
    // test02();
    // test03();
    test04();
}
