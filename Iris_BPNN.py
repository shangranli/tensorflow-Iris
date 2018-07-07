import tensorflow as tf
import input_orl
import input_iris
import numpy as np


INPUT_NODE = 4   # 输入节点
OUTPUT_NODE = 3    # 输出节点
LAYER1_NODE = 80   # 隐藏层节点数
LAYER2_NODE = 80
                              
BATCH_SIZE = 5      # 每次batch打包的样本个数        

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 20000        
MOVING_AVERAGE_DECAY = 0.99

train_size = 30
BATCH_SIZE = train_size
train_num = 3 * train_size
train_data,test_data,train_lab, test_lab = input_iris.read_iris(train_size)



#前向传播
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2, weights3, biases3):
    # 不使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)
        return tf.matmul(layer2, weights3) + biases3

    else:
        # 使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2))
        return tf.matmul(layer2, avg_class.average(weights3)) + avg_class.average(biases3)


def train():
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    # 生成隐藏层1的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成隐藏层1的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, LAYER2_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[LAYER2_NODE]))
    
    # 生成输出层的参数。
    weights3 = tf.Variable(tf.truncated_normal([LAYER2_NODE, OUTPUT_NODE], stddev=0.1))
    biases3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2, weights3, biases3)
    
    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2, weights3, biases3)
    
    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
    loss = cross_entropy_mean + regularaztion
    
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num/ BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
    
    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化会话，并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        test_feed = {x: test_data, y_: test_lab} 
        
        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict={x:test_data,y_:test_lab})
                print("After %d training step(s), validation accuracy using average model is %g "
                      % (i, validate_acc))
                total_cross_entropy =sess.run(loss,feed_dict={x:train_data,y_:train_lab})
                print("after %d training steps,cross entropy on all data is %g" %
                     (i,total_cross_entropy))
                #print(sess.run(average_y,feed_dict={x:train_data,y_:train_lab}))
            
            sess.run(train_op,feed_dict={x:train_data,y_:train_lab})
            
            
            
        '''test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))'''


def main(argv=None):
    train()

if __name__=='__main__':
    tf.app.run()
