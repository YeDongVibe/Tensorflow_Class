import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 함수
def f(x):
    return x**4 - 3*x**3 +2

def fprime(x):
    h = 0.001
    return (f(x+h) - f(x))/h

# 조건
k = 0 # 반복 횟수 초기화
mi = 0.001 # 반복 학습률
tol = 1e-5 # 허용오차 크기(종료조건)

xb = 0.0 # 변경 전 x값
xa = 4.0 # 변경 후 x값
xlist = [xa] # x 값 저장
x = tf.Variable(xb, dtype=tf.float32) # 변수 생성 및 초깃값 설정

while((xb-xa) > tol and k < mi):
    k += 1
    xb = x.numpy() # x 값을 Numpy 배열로 변환
    st = mi*fprime(x)
    x.assign_sub(st, read_value=False) # 값 업데이트(x값 읽지 않음)
    xa = x.numpy()
    xlist.append(xa)
    print('최종해', k, xa, f(xa)) # 최종 해 출력

    