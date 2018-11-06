import  tensorflow as tf
#  导入tensorflow包
import numpy as np

trX=np.linspace(-1,1,101)
# 使用numpy包中的linspace函数创建一个数组 从[1,100]范围内均分为101个
trY=2*trX+np.random.randn(*trX.shape)*0.23
# trY的值为2倍的trX的值加上一个随机产生的很小的噪声，其中大小为trX的维数大小，trX和TrY表示了一组train samples。

X=tf.placeholder("float")
Y=tf.placeholder("float")
# 这两行的意思是声明两个变量x,y,dtype="float"

def model(x,w):
    return tf.multiply(w,x)
# 定义了一个model函数，返回值为w*x;

w=tf.Variable(0.0,name="weight")
# 定义一个权重变量w 其初始值为0.0
y_model=model(X,w)
# 定义一个y_model 表示使用x值去求得的y值

cost=np.square(Y-y_model)
# 定义损失函数为平方差

train_op=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# 定义训练的方法为梯度下降法 其中学习率为0.01，目标是最小化cost的值

with tf.Session() as sess:
    # 在一个tensorflow会话中启动graph
    tf.global_variables_initializer().run()
    #添加用于初始化变量的节点

    for i in range(100):
        for (x, y) in zip(trX, trY):
            # zip(trX，trY)的含义是将其转换成[(trX1,trY1),(trX2,trY2)……]这样的形式
            sess.run(train_op, feed_dict={X: x, Y: y})
            #sess.run()表示在此次计算中并非运行整张图，而是为了fetch（取回）想要的部分去运行这一部分的内容。
    print(sess.run(w))









