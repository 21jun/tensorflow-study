## ML lab 03 - Linear Regression 의 cost 최소화의 TensorFlow 구현 (new)
[https://youtu.be/Y0EF9VqRuEA]

### Cost 시각화
```py
# Lab 3 Minimizing Cost
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.Session()

# Variables for plotting cost function
W_history = []
cost_history = []

for i in range(-30, 50):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict={W: curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

# Show the cost function
plt.plot(W_history, cost_history)
plt.show()
```
![img](img/lab03-01.png)


### 경사하강법 수식적용
```py
# Lab 3 Minimizing Cost
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data
# We know that W should be 1
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)      # 기울기(미분값)계산
descent = W - learning_rate * gradient          # 새로 적용될 W 값을 계산
update = W.assign(descent)                      # 텐서노드를 다시 assign 할떄는 assign 함수사용
                                                # W = W - learning_rate * gradient
# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(21):
    _, cost_val, W_val = sess.run([update, cost, W], feed_dict={X: x_data, Y: y_data})
    print(step, cost_val, W_val)
```
```py
 여기서 Minimize 부분을 아래의 이미 정의된 함수로 대체할수 있음 (옵티마이저) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)      
```
```py
 옵티마이저를 사용 할 때 수동으로 계산하는 경우처럼 gradient의 값을 조정하고 싶다면
 다음과 같은 방법을 이용함 
# Get gradients
gvs = optimizer.compute_gradients(cost, [W])
# 여기서 원한다면 gvs를 수정해서 apply 하면됨
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)
```