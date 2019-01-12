## ML lab 02 - TensorFlow로 간단한 linear regression을 구현 (new)
[https://youtu.be/mQGwjrStQgg]

```py
import tensorflow as tf

x_train = [1,2,3]
y_train = [1,2,3]

# Variable은 실행전에 초기화 해주어야함
# tf Variable은 텐서플로우가 스스로 값을 바꿀수 있는 노드임
W = tf.Variable(tf.random_normal([1]), name = 'weight') # shape이 [1]인 랜덤한 값
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Hypothesis
hypothesis = x_train * W + b

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

# Initializes global variables in the graph [7:8]
sess.run(tf.global_variables_initializer())

# Train data
for step in range(2001):
    sess.run(train)
    if step%20==0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
```
```
0 11.609702 [-0.86762834] [0.68823445]
20 0.3116761 [0.34482557] [1.1506397]
40 0.19029458 [0.48270783] [1.1436734]
...
1960 1.8342114e-05 [0.9950259] [0.01130747]
1980 1.6658809e-05 [0.99525964] [0.01077602]
2000 1.51294735e-05 [0.9954824] [0.01026957]
```
    lec02의 예상대로 W = 1 , b = 0 으로 수렴함


![img](img/lab02-01)

    train 노드는 cost, hypothesis, w, b 로 연결되어 있으므로 train만 run 하면 학습이 이뤄짐

```py
# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# Try to find values for W and b to compute Y = W * X + b
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# placeholders for a tensor that will be always fed using feed_dict
# See http://stackoverflow.com/questions/36693740/
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# Our hypothesis is X * W + b
hypothesis = X * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Learns best fit W:[ 1.],  b:[ 0]
'''
...
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
'''

# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

'''
[ 5.0110054]
[ 2.50091505]
[ 1.49687922  3.50495124]
'''


# Fit the line with new training data
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X: [1, 2, 3, 4, 5],
                            Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Learns best fit W:[ 1.],  b:[ 1.1]
'''
1980 2.90429e-07 [ 1.00034881] [ 1.09874094]
2000 2.5373e-07 [ 1.00032604] [ 1.09882331]
'''

# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

'''
[ 6.10045338]
[ 3.59963846]
[ 2.59931231  4.59996414]
'''

```

    placeholder를 이용하여 모델만 만들고 나중에 feed_dict로 값을 넘겨줄 수 있음
    (여러 값에따라 학습결과가 달라지게)

    학습된 모델을 테스트하려면 (Learns best fit W:[ 1.],  b:[ 1.1])
    print(sess.run(hypothesis, feed_dict={X: [5]})) # [ 6.10045338]
