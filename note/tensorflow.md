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

### argmax 함수

    list 중 가장큰 argument의 인덱스를 반환하는 함수

    첫번째 인자로 list를 , 두번째 인자로 axis 를 입력받는다(몇차원에서 계산할지)

```py
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
print(a, sess.run(tf.argmax(a, 1)))

# [[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]
```