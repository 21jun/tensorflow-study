### assign 함수
```py
W = tf.Variable(tf.random_normal([1]), name='weight')
descent = W - learning_rate * gradient
W.assign(descent)

텐서노드를 다시 할당할때에는 assign 함수를 사용해야함 (=)으로 할당하면 안됨
```

### reduce_mean 함수
```py
x = [[1.,1.], [2.,2.]]
a = tf.reduce_mean(x)
sess = tf.Session()
print(sess.run(a))          # 1.5

reduce_mean 함수는 차원을 모두 제거되고 단하나의 스칼라값(평균)을 리턴함
(1+1+2+2)/4 = 1.5

axis argument를 설정하여 몇차원까지 남기고 평균을 구할지 설정가능 (default = 0)

b = tf.reduce_mean(x, axis = 1) 
print(sess.run(b))          # [1. 2.]
```