import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow.python.keras as keras
from tensorflow.python.keras import layers
from sklearn.metrics import r2_score

# https://www.tensorflow.org/api_docs/python/tf/keras/datasets/boston_housing/load_data
# features 설명 (http://lib.stat.cmu.edu/datasets/boston)
# 보스톤 주택 가격(MEDV) 예측 (1인당 범죄율, 주택당 평균 방 개수, 학생대 교사 비율 등의 features 이용함)
# MEDV(주택 가격 중앙값, 단위: $1,000)

## x_train = 독립 변수 dataset , y_train = 종속 변수 dataset(MEDV(주택가격 중앙값))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path='boston_housing.npz', test_split=0.2, seed=113 
) # train_test_split ; 8:2
print('x_train.shape', 'y_train.shape', x_train.shape, y_train.shape)

## .hstack() = np array 수평으로 결합 -> 주어진 배열을 좌우로 연결해 새로운 배열 형성
all_train_data = np.hstack((x_train, y_train.reshape((-1, 1)))) #reshape() 활용해 2차원 배열로 변환(-1 ; 차원의 크기 자동지정)

## column name setting
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

## DataFrame으로 변환
df = pd.DataFrame(all_train_data, columns=column_names)
df.describe() # 기술 통계량 확인

## nomalization
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)
x_train -= x_mean
x_train /= x_std
x_test -= x_mean
x_test /= x_std

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
y_train -= y_mean
y_train /= y_std
y_test -= y_mean
y_test /= y_std

# model train
model = keras.Sequential([
    ## input layer(input_shape은 입력층에만 추가
    layers.Dense(units=16, activation='relu',input_shape=(13,)), ##input_shape ; 종속변수 하나를 제외한 feature의 갯수
    ## hidden layers
    layers.Dense(units=32, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    ## ouput layer(주택 가격을 하나의 값으로 예측하므로 뉴런수 = 1로 지정)
    layers.Dense(units=1) ## 출력층에 활성화 함수를 따로 지정하지 않으면 linear 함수가 적용(회귀문제에서 사용되는 일반적인 방법)
    ])

# model compile
# optimizer : 모델의 가중치 업데이트 방법 정의, loss : 모델의 성능을 평가, metrics : 모델의 성능을 평가하는데 사용되는 metric의 리스트)
model.compile(optimizer='adam', loss='mse')
# model.summary()

# model fitting
# (feature_data, target_data, epochs(전체 데이터셋을 학습하는데 사용되는 에포크 수),batch_size(한 번에 처리되는 샘플의 갯수))
history = model.fit(x_train, y_train, epochs=50, batch_size=5, validation_split=0.2)


y_pred_train = model.predict(x_train) * y_std + y_mean
y_pred_test = model.predict(x_test) * y_std + y_mean

## model 정확도 확인
# R-squared 계산
train_r2 = r2_score(y_train * y_std + y_mean, y_pred_train)
test_r2 = r2_score(y_test * y_std + y_mean, y_pred_test)

## 정확도 출력
print("Train 정확도: ", train_r2)
print("Test 정확도 : ", test_r2)

# loss func
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()

# 집값 예측 그래프
plt.figure(figsize=(12, 6))
plt.scatter(y_train * y_std + y_mean, y_pred_train, label='Train', color='blue')
plt.scatter(y_test * y_std + y_mean, y_pred_test, label='Test', color='orange')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Housing Price Prediction')
plt.legend()
plt.show()
# sns.pairplot(df[['CRIM', 'ZN', 'INDUS', 'TAX', 'MEDV']], diag_kind='kde')