## ML lec9-1: XOR 문제 딥러닝으로 풀기
[https://youtu.be/GYecDQQwTdI]

### XOR 문제 해결

    (0, 0) -> 0
    (0, 1) -> 1
    (1, 0) -> 1
    (1, 1) -> 0 
    
    하나의 Logistic Regression 으로는 해결할 수 없다. (수학적으로 증명됨)

    Multiple Logistic Regression 으로는 해결 할 수 있다.

![img](img/lec09-01.png)

    (x1, x2) 입력값이 들어올때 2개의 다른 Logistic Regression 에 넣어서 
    얻어낸 각각의 결과값을 (y1, y2) 라고 하고, 
    이를 다시 새로운(3rd) Logistic Regression 에 집어 넣으면 XOR 문제를 해결할 수 있다

> 앞선 강의에서 (lec6) 여러개의 Logistic Regression 으로 Multinomial Classification 을 만듬

    Multinomial Classification 의 식을 계산하기 편하게 행렬식으로 만들었었음

    이번에도 마찬가지로 여러개의 Logistic Regression 을 사용하므로 같은 방식으로 행렬식 만듬

![img](img/lec09-02.png)

    수식은 다음과 같음 (K = 첫 2개의 로지스틱회귀 거침)
    H(x) = sigmoid(K(x)*W2 + b2)    여기서 세번째 모델 거침
