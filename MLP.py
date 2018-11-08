import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess=tf.InteractiveSession()
#  创建一个默认的session

in_units=784 #输入节点数
h1_units=300 #隐含层的输出节点数
W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
# 设置weights，将权重初始化为截断的正态分布,从这其中产生随机值，其值范围为（μ-2σ，μ+2σ），且其shape=[in_units,h1_units],标准差为0.1
b1=tf.Variable(tf.zeros([h1_units]))
W2=tf.Variable(tf.zeros([h1_units,10]))
b2=tf.Variable(tf.zeros([10]))

x=tf.placeholder(tf.float32,[None,in_units])# 输入
keep_prob=tf.placeholder(tf.float32)#dropout的比率 即保留数据而不置为0的比率 主要的功能是防止过拟合

hidden1=tf.nn.relu(tf.matmul(x,W1)+b1) #使用relu进行激活
hidden1_drop=tf.nn.dropout(hidden1,keep_prob) #随机将一部分节点置为0
y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)  #输出层

y_=tf.placeholder(tf.float32,[None,10])

#以上已经定义完算法公式，即nn forward时候的计算

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
#使用cross_entropy表示损失

train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)


tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})
#     在训练集上每次取100个数据进行训练，并且在隐含层保留0.75的数据

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
# 在测试集对训练得到的模型进行测试 输出最终额准确率


#  在多层感知机上对测试集进行测试最终达到近似98%的正确率，由于原来的不使用隐含层的线性回归方法
