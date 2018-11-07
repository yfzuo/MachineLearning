# 目标是实现去噪自编码器。即用少量的稀疏的高阶特征对输入进行重构（这里的输入是图像），重构的结果应该和原有的图像相同。
# 使用了一层隐含层
import  tensorflow as tf
import  numpy as np
import  sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def xavier_init(fan_in,fan_out,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)
# 构造一个标准的均匀分布的Xaiver初始化器，fan_in 是输入节点的数量，fan_out是输出节点的数量
# xavier会构造出一个均值为0，但是方差为2/(n_in+n_out)的分布。其中分布可以采用均匀分布也可以采用高斯分布

class AdditiveGuassianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input#输入变量数
        self.n_hidden=n_hidden #隐含节点数
        self.transfer=transfer_function #隐含层激活函数 其中默认为softplus。softplus(x)=log(1+exp(x)),是sigmoid的原函数，其导数为sigmoid函数
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale#高斯噪声系数 默认为0.1
        network_weight=self._initialize_weights()
        self.weights=network_weight

        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)),
                                                   self.weights['w1']),self.weights['b1']))
        # hidden的含义是从原有的输入中提取相应的高阶特征的隐含层
        # 具体的操作是在原有基础上加入噪声，再与权重做矩阵相乘操作，使用add函数给得到的结果加上一个偏置，最后使用transfer去激活隐含层

        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        # reconstruction 是对隐含层得到的高阶特征进行重构。reconstruction=w2*hidden+b2

        self.cost=0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        # 定义自编码器的损失函数，使用square error作为cost.

        self.optimizer=optimizer.minimize(self.cost)
        # 训练的目标是求得最小的cost

        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights=dict()#初始化一个字典
        all_weights['w1']=tf.Variable(xavier_init(self.n_input,self.n_hidden))#w1和b1是隐含层的参数。
        all_weights['b1']=tf.Variable(tf.zeros(self.n_hidden),dtype=tf.float32)
        all_weights['w2']=tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype=tf.float32))#w2 b2由于没有使用激活函数，所以都初始化为0
        all_weights['b2']=tf.Variable(tf.zeros([self.n_input]),dtype=tf.float32)
        return all_weights

    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),
                               feed_dict={self.x: X,self.scale:self.training_scale})
    #     我们已经定义了损失函数cost，这里是代入参数值调用相应的函数。
        return cost

    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
    # 单纯计算cost的函数，但是在这里并不进行训练。用于在训练完毕之后，在测试集上进行测评

    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    # 触发隐含层的节点，返回的是隐含层的输出结果

    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size = self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    # 这里使用隐含层的输出作为输入进行重构

    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
    # 可以看出transfer和generate构成了这个自编码器的两个部分，这里是对所有的步骤执行一遍

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBias(self):
        return self.sess.run(self.weights['b1'])

def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test
#定义一个对训练和测试数据进行标准化处理的函数，即让数据变成均值为0,标准差为1的分布。这里使用sklearn.preprocessing上的类，
# 但是需要使用同一个preprocessor进行处理训练数据和测试数据


def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]
#     采用随机不放回的抽样形式抽取batch_size大小的数据

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
# 对训练数据和测试数据进行标准化

n_samples=int(mnist.train.num_examples)
training_epochs=20#训练轮数为20
batch_size=128
display_step=1

autoencoder=AdditiveGuassianNoiseAutoencoder(n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
                                             optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)
# 实例化一个autoencoder

for epoch in range(training_epochs):
    avg_cost=0
    total_batch=int(n_samples/batch_size)
    for i in range(total_batch):
        batch_xs=get_random_block_from_data(X_train,batch_size)

        cost=autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
    #  在每一轮中对每一个batch求cost并累加总共的batch

    if epoch % display_step==0:
        print("Epoch:",'%04d' %(epoch+1),"cost=", "{:.9f}".format(avg_cost))

    print("Total cost: "+str(autoencoder.calc_total_cost(X_test)))