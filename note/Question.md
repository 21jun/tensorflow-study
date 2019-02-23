## Q:soft max 적용

    Softmax 함수는 logits(hypothesis)에 적용하지 않고(tf.nn.softmax())
    cost function 에만 적용하는게 맞는가?
    (cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)))

### 예시코드

```py
# 기존의 코드 (softmax 적용..)

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

```

```py
# ReLU를 이용한 코드... softmax 미적용

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

```

## A: softmax_cross_entropy_with_logits 을 사용하면 softmax가 적용되는것과 같다
    
    인자로 softmax가 적용되기전의 logits을 요구하기 때문
```py
    logits = tf.matmul(L2, W3) + b3
    # ogits = 예측값 (softmax를 거치기전의 Hypothesis)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
```
    결국 마지막 노드인 optimizer를 학습시킬때에는 Hypothesis가 아닌 cost를 사용하므로
    (cost는 Hypothesis와 연결되어있음)
    cost 를 softmax_cross_entropy_with_logits 으로 적용하려면 softmax를 거치지 않은 logits 를 넣어줌
    (cost와 Hypothesis가 아닌 logits과 연결)

    결국 같은거임
    softmax_cross_entropy_with_logits 을 사용하지 않고 수식으로 구현한다면,

    hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

    같이 구현해야 할것 (logits에 softmax를 적용한 hypothesis)

## CNN 에서 필터의 크기

    필터의 크기를 5x5, 7x7로 하는 것보다, 3x3 필터를 여러개 사용하는것이 계산량이 적음

    그렇다면 가장빠른 1x1 필터를 사용하지 않는이유는?
    
    [https://iamaaditya.github.io/2016/03/one-by-one-convolution/]

![img](img/cnn_1x1.png)

    CS231n [https://youtu.be/pA4BsUK3oP4?t=1974]

    Most simplistic explanation would be that 1x1 convolution leads to dimension reductionality. 
    For example, an image of 200 x 200 with 50 features on convolution with 20 filters of 1x1 would
    result in size of 200 x 200 x 20. But then again, is this is the best way to do dimensionality
    reduction in the convoluational neural network? What about the efficacy vs efficiency?

    1x1 필터는 인풋 이미지의 픽셀을 그대로 옮기지만, 각 픽셀의 feature를 1개로 줄이기때문에
    200x200x50 의 인풋이 20개의 1x1 필터를 지나면, 200x200x20 의 정보만 남게됨..
    이것은 차원을 줄이기 때문에 정보의 손실이 있음
