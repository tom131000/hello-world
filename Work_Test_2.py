import tensorflow as tf
import numpy as np
import random
import pandas as pd

class ToySequenceData(object):
    def __init__(self, n_samples=1000, begin = 0, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            len = 29
            temp_list = []
            self.seqlen.append(len)
            for j in range(29):
                temp_list.append([data[i + begin][j]])
            self.data.append(temp_list)
            self.labels.append(label[i + begin])
        self.batch_id = 0

    def next(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen

f = open('Log_Data.csv')
f2 = open('mlabel.csv')
df = pd.read_csv(f)
df2 = pd.read_csv(f2)
data = df.iloc[:,0:29].values
label = df2.iloc[:,0:2].values


learning_rate = 0.01
training_iters = 1000000
batch_size = 1280
display_step = 10
seq_max_len = 29
n_hidden = 64
n_classes = 2
trainset = ToySequenceData(n_samples=460111, max_seq_len=seq_max_len)
testset = ToySequenceData(n_samples=115027, begin= 460111, max_seq_len=seq_max_len)

x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
seqlen = tf.placeholder(tf.int32, [None])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def dynamicRNN(x, seqlen, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, 1])
    x = tf.split(axis=0, num_or_size_splits=seq_max_len, value=x)
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)
    print(outputs)
    outputs = tf.stack(outputs)
    print(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    print(outputs)
    batch_size = tf.shape(outputs)[0]
    print(batch_size)#128
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)#?
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    print(outputs)
    return tf.matmul(outputs, weights['out']) + biases['out']


pred = dynamicRNN(x, seqlen, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
negative_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
positvie_pred = tf.equal(tf.argmin(pred,1), tf.argmin(y,1))
accuracy = tf.reduce_mean(tf.cast(negative_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))