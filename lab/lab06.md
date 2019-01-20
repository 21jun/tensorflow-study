## ML lab 06-1: TensorFlow로 Softmax Classification의 구현하기
[https://youtu.be/VRnubDzIy3A]

```py
# Lab 6 Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# X 데이터는 4가지 feature를 가짐
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]

# Y (정답)은 3가지중 1개 이므로 다음과 같이 표기함 = One Hot Encoding
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3  # 분류할 class 의 가짓수

# 4가지 feature가 들어오고 3개의 클래스로 분류하므로 shape=[4,3]
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b) # softmax 함수를 적용한 H

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))   # Cross entropy

# GD 를 이용하여 학습
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
            _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: y_data})

            if step % 200 == 0:
                print(step, cost_val)

    print('--------------')
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))

'''
0 6.926112
200 0.6005015
400 0.47295815
600 0.37342924
800 0.28018373
1000 0.23280522
1200 0.21065344
1400 0.19229904
1600 0.17682323
1800 0.16359556
2000 0.15216158
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]
-------------
[[0.9311919  0.06290216 0.00590591]] [0]
-------------
[[1.2732815e-08 3.3411323e-04 9.9966586e-01]] [2]
-------------
[[1.3890490e-03 9.9860185e-01 9.0613084e-06]
 [9.3119192e-01 6.2902197e-02 5.9059085e-03]
 [1.2732815e-08 3.3411323e-04 9.9966586e-01]] [1 0 2]
'''
```

## ML lab 06-2: TensorFlow로 Fancy Softmax Classification의 구현하기
[https://youtu.be/E-io76NlsqA]

```py
# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

'''
(101, 16) (101, 1)
'''

nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

# 지금 입력데이터는 one hot이 아니고 0~6의 숫자로 되어있음
# [0,0,0,0,0,0,1] 같은 one hot 으로 변경시켜야함
Y_one_hot = tf.one_hot(Y, nb_classes)   # one hot
print("one_hot:", Y_one_hot)            # one hot을 적용하면 rank가 1증가함
# ex) [[1],[3]] -> [[[0,2,0,0,0,0,0]],[[0,0,0,1,0,0,0]]]

# 우리가 원하는 모양으로 reshape 해야함 
# [[[0,2,0,0,0,0,0]],[[0,0,0,1,0,0,0]]] -> [[0,2,0,0,0,0,0],[0,0,0,1,0,0,0]]
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # -1의 의미는 None처럼 갯수를 미정하는것
print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={X: x_data, Y: y_data})
                                        
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
...
Step:  1800	Loss: 0.060	Acc: 100.00%
Step:  1900	Loss: 0.057	Acc: 100.00%
Step:  2000	Loss: 0.054	Acc: 100.00%
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 6 True Y: 6
[True] Prediction: 1 True Y: 1
'''
```

```py
# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b) # softmax 함수를 적용한 H

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))   # Cross entropy
```
    기존에 수식으로 적었던 softmax cost function을 아래와 같이 하나의 함수로 표현가능 
```py
# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                 labels=tf.stop_gradient([Y_one_hot])))
```
    logits = 예측값 (softmax를 거치기전의 Hypothesis)

```py
# 지금 입력데이터는 one hot이 아니고 0~6의 숫자로 되어있음
# [0,0,0,0,0,0,1] 같은 one hot 으로 변경시켜야함
Y_one_hot = tf.one_hot(Y, nb_classes)   # one hot
print("one_hot:", Y_one_hot)            # one hot을 적용하면 rank가 1증가함
# ex) [[1],[3]] -> [[[0,2,0,0,0,0,0]],[[0,0,0,1,0,0,0]]]

# 우리가 원하는 모양으로 reshape 해야함 
# [[[0,2,0,0,0,0,0]],[[0,0,0,1,0,0,0]]] -> [[0,2,0,0,0,0,0],[0,0,0,1,0,0,0]]
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # -1의 의미는 None처럼 갯수를 미정하는것
print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 7), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 7), dtype=float32)
'''
```