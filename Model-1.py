import tensorflow as tf
import numpy as np
import pandas as pd

learning_rate = 0.001
training_epochs = 10000

#########################数据准备###############################################
#录入所有原始数据
x_all = pd.read_csv('feed_x.csv')
y_all = pd.read_csv('feed_y.csv') 

#转换为np数组格式
data_x = np.zeros([19220,18])
data_x = x_all.values

data_y = np.zeros([19220,2]) 
data_y = y_all.values

#切分数组为三个数据集比例为6：2：2
x_input = data_x[0:11500,:]
y_output = data_y[0:11500,:]

x_v = data_x[11500:15500,:]
y_v = data_y[11500:15500,:]

x_t = data_x[15500:,:]
y_t = data_y[15500:,:]

###########################定义神经网络#########################################
def addlayer(inputdata,input_size,out_size,active=None):
    weights=tf.Variable(tf.random_normal([input_size,out_size]))
    bias=tf.Variable(tf.zeros([1,out_size])+0.1)
    wx_plus_b=tf.matmul(inputdata,weights)+bias
    if active==None: 
        return wx_plus_b,weights 
    else: 
        return active(wx_plus_b),weights

###############################训练数据占位符########################################################
xinput=tf.placeholder(tf.float32,[None,18])
youtput=tf.placeholder(tf.float32,[None,2])

###########################定义三层神经网络 18->10->2#################################
layer1 = addlayer(xinput,18,10,tf.nn.tanh)
prediction = addlayer(layer1[0],10,2,active=tf.nn.softmax)

###########################定义损失函数#############################################
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=youtput, logits=prediction[0]+1e-09))  

###########################定义训练目标#############################################
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
 
#################变量初始化########################################################
init=tf.global_variables_initializer()

###########################进行神经网络训练########################################
correct_prediction = tf.equal(tf.argmax(prediction[0],1), tf.argmax(y_output,1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epochs):  
        if i%10 == 0:  
            sess.run(train, feed_dict={xinput:x_input, youtput:y_output})  
            train_accuracy = accuracy.eval(feed_dict={xinput:x_input, youtput:y_output})  
            lossvalue = loss.eval(feed_dict={xinput:x_input, youtput:y_output}) 
            print(lossvalue,"step %d, training accuracy %g"%(i, train_accuracy))  
############################权重打印######################################################################
    layer1result = sess.run(layer1[1], feed_dict={xinput:x_input, youtput:y_output})   #首层权重
    print(layer1result)  
    predictionresult = sess.run(prediction[1],feed_dict={xinput:x_input, youtput:y_output})#中间层权重
    print(predictionresult) 





