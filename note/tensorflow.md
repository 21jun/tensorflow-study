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

axis argument를 설정하여 몇차원까지 남기고 평균을 구할지 설정가능 (default = 모든 값을 평균내라)

b = tf.reduce_mean(x, axis = 1) 
print(sess.run(b))          # [1. 2.]
```

### argmax 함수

    list 중 가장큰 argument의 [인덱스]를 반환하는 함수

    첫번째 인자로 list를 , 두번째 인자로 axis 를 입력받는다(몇차원에서 계산할지)

    one-hot 인코딩되어있는 데이터셋에서 사용하면 됨
```py
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
print(a, sess.run(tf.argmax(a, 1)))

# [[1.3890490e-03 9.9860185e-01 9.0613084e-06]] [1]
```

### reshape 함수

    텐서의 shape을 바꿔주는 함수

    shape=[] 으로 shape을 지정해주면 해당 shape으로 바꿔준다

    보통 제일 안쪽 shape은 건들지 않는다.. 예시에서는 3

    맨앞에 -1을 적어주면 자동으로 계산된 크기의 shape으로 바꾼다

```py
t = np.array([[[0, 1, 2], 
               [3, 4, 5]],
              
              [[6, 7, 8], 
               [9, 10, 11]]])
t.shape # (2, 2, 3)

tf.reshape(t, shape=[-1, 3]).eval()
'''
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
'''
```   
> squeeze

    텐서의 차원을 1단계 줄여준다
```py
tf.squeeze([[0], [1], [2]]).eval()
# array([0, 1, 2], dtype=int32)
```

> expand_dims

    얼마나 차원을 늘릴지 지정해주면 텐서의 차원을 늘려준다. (예시는 1만큼늘림)
```py
tf.expand_dims([0, 1, 2], 1).eval()
'''
array([[0],
       [1],
       [2]], dtype=int32)
'''
```

### one_hot 함수

    랭크가 1증가하게됨..

    추가


### eval 함수

    sses.run() 과 같은 효과로, 텐서뒤에 .eval() 으로 실행시킨다.
    session=sess 로 세션을 지정해줄수도 있다.
    [https://stackoverflow.com/questions/33610685/in-tensorflow-what-is-the-difference-between-session-run-and-tensor-eval]

    차이점은 eval은 한개의 텐서를 실행시키는 것이고,
    Session.run 은 한번에 여러개의 텐서들을 넣어 실행시킬 수 있다.

```py
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
```