#include "../include/net.hpp"
#include "../include/utils.hpp"
#include <chrono>
using namespace std;


// 网络基本功能
void test01()
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count(); // 从epoch开始经过的纳秒数做种子
    default_random_engine generator(seed); // 随机数生成引擎
    normal_distribution<_Float32> dist(0.,1.);
    vector<int> layers;
    layers.push_back(3);
    layers.push_back(4);
    layers.push_back(2);
    net<_Float32> mlp(layers);
    int k = 0;
    for (auto i = mlp.weights.begin(); i != mlp.weights.end(); i++)
    {
        cout << "weights " << k  << ": " << endl;
        Matrix<_Float32>::printMatrix(*i);
        k++; 
    }
    k = 0;
    for (auto i = mlp.bias.begin(); i != mlp.bias.end(); i++)
    {
        cout << "bias " << k  << ": " << endl;
        Matrix<_Float32>::printMatrix(*i);
        k++; 
    }

    Matrix<_Float32> data(10,3);
    for (size_t i = 0; i < data.row; i++)
    {
        for (size_t j = 0; j < data.col; j++)
        {
            data.ma[i][j] = dist(generator);
        }
    }
    cout << "data: " << endl;
    Matrix<_Float32>::printMatrix(data);
    Matrix<unsigned> label(10,1);
    Matrix<_Float32> z_last = mlp.fp(data);
    cout << "z_last: " << endl;
    Matrix<_Float32>::printMatrix(z_last);
    cout << "loss: " << endl;
    Matrix<_Float32>::printMatrix(cross_entrpy(z_last,label));
    pair<vector<Matrix<_Float32>>,vector<Matrix<_Float32>>> grad = mlp.bp(z_last,label);
    k = 0;
    
    for (auto i = grad.first.begin(); i != grad.first.end(); i++)
    {
        cout << "grad_b " << k  << ": " << endl;
        Matrix<_Float32>::printMatrix(*i);
        cout << endl;
        k++;
    }
    k = 0;
    
    for (auto i = grad.second.begin(); i != grad.second.end(); i++)
    {
        cout << "grad_w " << k  << ": " << endl;
        Matrix<_Float32>::printMatrix(*i);
        cout << endl;
        k++;
    }

    mlp.step(grad,0.1);
    cout << "after updates: " << endl;
    k = 0;
    for (auto i = mlp.weights.begin(); i != mlp.weights.end(); i++)
    {
        cout << "weights " << k  << ": " << endl;
        Matrix<_Float32>::printMatrix(*i);
        k++; 
    }
    k = 0;
    for (auto i = mlp.bias.begin(); i != mlp.bias.end(); i++)
    {
        cout << "bias " << k  << ": " << endl;
        Matrix<_Float32>::printMatrix(*i);
        k++; 
    }
}

// 小demo测试
void test02()
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count(); // 从epoch开始经过的纳秒数做种子
    default_random_engine generator(seed); // 随机数生成引擎
    normal_distribution<_Float32> dist(0.,1.);
    vector<int> layers;
    layers.push_back(3);
    layers.push_back(4);
    layers.push_back(2);
    net<_Float32> mlp(layers);
    Matrix<_Float32> data(10,3);
    for (size_t i = 0; i < data.row; i++)
    {
        for (size_t j = 0; j < data.col; j++)
        {
            data.ma[i][j] = dist(generator);
        }
    }
    Matrix<unsigned> label(10,1);
    for (size_t i = 0; i < label.row; i++)
    {
        if (i % 2 == 0)
        {
            label.ma[i][0] = 1;
        }
        
    }
    // train
    for (size_t i = 0; i < 200; i++)
    {   
        Matrix<_Float32> z_last = mlp.fp(data);
        if (i % 10 == 0)
        {
            cout << "loss: " << endl;
            Matrix<_Float32>::printMatrix(cross_entrpy(z_last,label));
            Matrix<unsigned> res(z_last.row,1);
            for (size_t i = 0; i < z_last.row; i++)
            {
                _Float32 max = z_last.ma[i][0];
                unsigned index = 0;
                for (size_t j = 0; j < z_last.col; j++)
                {
                    if (z_last.ma[i][j] > max)
                    {
                        max = z_last.ma[i][j];
                        index = j;
                    }
                    res.ma[i][0] = index;

                }

            }
            _Float32 acc = 0.;
            for (size_t i = 0; i < res.row; i++)
            {
                if (res.ma[i][0] == label.ma[i][0])
                {
                    acc ++;
                }
            }
            cout << "acc: " << acc / res.row << endl;
        }
        
        pair<vector<Matrix<_Float32>>,vector<Matrix<_Float32>>> grad = mlp.bp(z_last,label);
        mlp.step(grad,0.1);
    }
    // test
    cout << "testing:" << endl;
    Matrix<_Float32> z_last = mlp.fp(data);
    Matrix<unsigned> res(z_last.row,1);
    for (size_t i = 0; i < z_last.row; i++)
    {
        _Float32 max = z_last.ma[i][0];
        unsigned index = 0;
        for (size_t j = 0; j < z_last.col; j++)
        {
            if (z_last.ma[i][j] > max)
            {
                max = z_last.ma[i][j];
                index = j;
            }
            res.ma[i][0] = index;
            
        }
        
    }
    cout << "res: " << endl;
    Matrix<unsigned>::printMatrix(res);
    cout << "label: " << endl;
    Matrix<unsigned>::printMatrix(label);
    _Float32 acc = 0.;
    for (size_t i = 0; i < res.row; i++)
    {
        if (res.ma[i][0] == label.ma[i][0])
        {
            acc ++;
        }
    }
    cout << "acc: " << acc / res.row << endl;
    
    
}

/**
 * @brief 验证网络在测试集上的性能
 * 
 * @tparam dataType1 
 * @tparam dataType2 
 * @param mlp 
 * @param images 
 * @param labels 
 */
template<class dataType1, class dataType2>
void evaluate(net<dataType1> &mlp, Matrix<dataType1> &images, Matrix<dataType2> &labels);

/**
 * @brief 训练网络
 * 
 * @tparam dataType1 
 * @tparam dataType2 
 * @param mlp 
 * @param images 
 * @param labels 
 * @param epochs 
 * @param batch_size 
 * @param lr 
 */
template<class dataType1, class dataType2>
void MBSD_train_test(net<dataType1> &mlp, Matrix<dataType1> &images, Matrix<dataType2> &labels, int epochs, int batch_size, _Float32 lr, Matrix<dataType1> &test_imgs, Matrix<dataType2> &test_labels);

// 测试
int main()
{
    // test01();
    // test02();

    // 1. 加载数据集
    Matrix<unsigned> Y_train(60000,1);
    load_mnist_label("../data/MNIST/train-labels-idx1-ubyte",Y_train);
    Matrix<_Float32> X_train(60000,784);
    load_mnist_images("../data/MNIST/train-images-idx3-ubyte",X_train);
    Matrix<unsigned> Y_test(10000,1);
    load_mnist_label("../data/MNIST/t10k-labels-idx1-ubyte",Y_test);
    Matrix<_Float32> X_test(10000,784);
    load_mnist_images("../data/MNIST/t10k-images-idx3-ubyte",X_test);
    // 2. 数据集预处理
    X_train.normalize();
    X_test.normalize();
    // 3. 定义模型参数
    vector<int> layers;
    layers.push_back(784);
    layers.push_back(512);
    layers.push_back(10);
    net<_Float32> mlp(layers);
    // 4. 定义超参数
    int epochs = 20;
    int batch_size = 500;
    _Float32 lr = 0.01;
    // 5. 调用训练函数
    MBSD_train_test(mlp,X_train,Y_train,epochs,batch_size,lr,X_test,Y_test);
    
}

template<class dataType1, class dataType2>
void evaluate(net<dataType1> &mlp, Matrix<dataType1> &images, Matrix<dataType2> &labels)
{
    mlp.set_val();
    Matrix<dataType1> z_last = mlp.fp(images);
    Matrix<dataType2> predict = z_last.argmax(0); // 取出每行最大值的下标
    assert(predict.row == labels.row);
    _Float32 acc = 0.;
    for (size_t i = 0; i < predict.row; i++)
    {
        if (predict.ma[i][0] == labels.ma[i][0])
        {
            acc ++;
        }
    }
    acc = acc / labels.row;
    cout << "accuracy in testset: " << acc << endl;
    
}

template<class dataType1, class dataType2>
void MBSD_train_test(net<dataType1> &mlp, Matrix<dataType1> &images, Matrix<dataType2> &labels, int epochs, int batch_size, _Float32 lr, Matrix<dataType1> &test_imgs, Matrix<dataType2> &test_labels)
{
    unsigned seed = chrono::system_clock::now().time_since_epoch().count(); // 从epoch开始经过的纳秒数做种子
    srand(seed);
    int dataset_size = images.row;
    vector<int> permutation;
    for (size_t i = 0; i < dataset_size; i++)
    {
        permutation.push_back(i);
    }
    for (size_t i = 0; i < epochs; i++)
    {
        // 随机打乱数据集和标签
        cout << "shuffle the dataset..." << endl;
        random_shuffle(permutation.begin(),permutation.end());
        Matrix<dataType1> tmp_images(images.row,images.col);
        Matrix<dataType2> tmp_labels(images.row,1);
        for (size_t i = 0; i < dataset_size; i++)
        {
            // 复制打乱的数据
            tmp_images.ma[i] = images.ma[permutation[i]]; 
            tmp_labels.ma[i] = labels.ma[permutation[i]];
        }
        images.ma.swap(tmp_images.ma);
        labels.ma.swap(tmp_labels.ma);
        // 清除 tmp占用的空间
        tmp_images.ma.resize(0);
        vector<vector<dataType1>> (tmp_images.ma).swap(tmp_images.ma);
        tmp_labels.ma.resize(0);
        vector<vector<dataType2>> (tmp_labels.ma).swap(tmp_labels.ma);

        // 准备mini_batch数据
        vector<Matrix<dataType1>> X_batches;
        vector<Matrix<dataType2>> Y_batches;
        for (size_t j = 0; j < dataset_size; j++)
        {
            if (j*batch_size + batch_size <= dataset_size)
            {
                Matrix<dataType1> x_batch(batch_size,images.col);
                Matrix<dataType2> y_batch(batch_size,1);
                x_batch.ma.assign(images.ma.begin() + j * batch_size, images.ma.begin() + (j + 1) * batch_size);
                y_batch.ma.assign(labels.ma.begin() + j * batch_size, labels.ma.begin() + (j + 1) * batch_size);

                X_batches.push_back(x_batch);
                Y_batches.push_back(y_batch);
            }
        }
        // 开始训练
        vector<dataType1> L;
        mlp.set_train();
        for (size_t j = 0; j < X_batches.size(); j++)
        {
            Matrix<dataType1> z_last = mlp.fp(X_batches[j]);
            L.push_back(cross_entrpy(z_last,Y_batches[j]).ma[0][0]);
            pair<vector<Matrix<_Float32>>,vector<Matrix<_Float32>>> grad = mlp.bp(z_last,Y_batches[j]);
            mlp.step(grad,lr);
        }
        dataType1 mean_loss = 0.;
        for (auto it = L.begin(); it != L.end(); it++)
        {
            mean_loss += *(it);
        }
        mean_loss = mean_loss / X_batches.size();
        cout << "epoch " << i << " loss: " << mean_loss << endl;
        if ( (i + 1) % 2 == 0)
        {
            cout << "testing..." << endl;
            evaluate(mlp,test_imgs,test_labels);

        }
    }
}
