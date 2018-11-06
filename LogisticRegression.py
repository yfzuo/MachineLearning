import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 这段代码主要做的事情是识别手写的数字（digit recognization）,mnist这个数据集中，train_set有55000个样本，
# test_set包含了10000个样本，validate_set中包含了5000个样本，其中每一个样本是28pi×28pi的灰度图片。
# 在训练中，将每个样本中的28×28的矩阵展开成1维向量。而对于输出的种类（0,1,2……）则也使用一个1维向量表示，
# 对于每个样本的label使用one-hot 编码。

mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
# 读入数据集

sess =tf.InteractiveSession()#将这个session注册为默认的session，并且对于之后的运算也在这个session中执行。

x=tf.placeholder(tf.float32,[None,784])
#如上文所述，将每个样本抽象成一个1维的向量，28×28=784.
# 创建一个placeholder, 用来输入数据，其中数据的类型是float32,none表示的是不限制条数，但是每条输入是一个784维的向量

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
# 此两行创建的是模型的weight和bias，其中对于weight而言，784为特征的维数，而10 为one-hot编码的10类。

y=tf.nn.softmax(tf.matmul(x,W)+b)
# softmax(x_i)=exp(x_i)/(sum_j(exp(x_j)))其含义是求得每个exp(x_i),再标准化。

y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
# 接下来定义一个损失函数，使用cross_entropy  为-sum（真实值×log（预测值））mean是对每个batch 求平均

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 定义训练操作，采用梯度下降的方式，以0.5的学习率求解最小的cross_entropy

tf.global_variables_initializer().run()
# 使用全局参数初始化器。

# 迭代地执行训练操作train_step
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    # 以100个样本为一个batch
    train_step.run({x:batch_xs,y_:batch_ys})

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# tf.argmax(y,1)的含义是求出预测的每个数字中最大的那个，correct_prediction是一个bool值，表示预测是否正确，

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#将correct_prediction转换成float32,统计所有样本的准确值。

print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
# 在测试集上进行测试，并输出其准确率。



