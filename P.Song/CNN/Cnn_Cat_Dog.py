from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 모델 훈련
model = models.Sequential()
# 필터수(32)
model.add(layers.Conv2D(32, (3, 3), activation= 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D(2,2)) # 최대 풀링 연산 적용할 윈도우 사이즈(다운샘플링:크기 축소)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
# 여기까지 Convolution 기반 층(지역 패턴 추출 층)

# 완전 연결 층 (전역 패턴 추출, 분류기)
model.add(layers.Flatten()) # 결과 1차원 백터로 변환
model.add(layers.Dense(512, activation='relu')) # 512차원 벡터공간에 투영
model.add(layers.Dense(1, activation='sigmoid')) # 2진 데이터라서 시그모이드 사용

model.summary()
''' 1. 150*150 입력 이미지에서 3*3 윈도우 슬라이딩하면서 3*3 패치 추출 -> 32개 필터에 대해 합성곱 -> 148*148*32
    2. 2*2 윈도우 1의 출력 특성 맵에 적용해서 패치 구역 별 최댓값만 추출 -> 출력 특성 맵 크기 절반으로 감소 -> 74*74*32
    3. 2의 출력 특성 맵에서 다시 3*3 패치 추출 -> 64개 필터에 대해 합성곱 -> 72*72*64
    4. 2처럼 최대 풀링 연산 3 출력에 적용 -> 출력 특성 맵 크기 절반으로 감소 -> 36*36*64
    5. 3*3 패치, 128개 필터에 대해 합성곱 -> 34*34*128
    6. 최대 풀링 연산 적용 -> 17*17*128
    7. 3*3 패치, 128개 필터에 대해 합성곱 -> 15*15*128
    8. 초대 풀링 연산 적용 -> 7*7*128
    9. 완전 연결 분류기 주입 위해 1차원 텐서(벡터)로 변환하는 층
    10. 5102차원 벡터공간에 투영
    11. 1차원 벡터공간으로 차원축소 후 시그모이드 함수 적용
'''
# 모델 컴파일
opt = optimizers.adam.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1./255) # 스케일 1/255로 조정. 부동소수점 형태로 변환
test_datagen = ImageDataGenerator(rescale=1./255) # 스케일 조정

train_generator = train_datagen.flow_from_directory(
    "C:/Ye_Dong/Tensorflow_Class/P.Song/CNN/CnnData/train", 
    target_size=(150, 150), # 네트워크 입력 규격에 맞게 크기 변환
    batch_size=20, # 1에폭 동안 투입 할 데이터 묶음
    class_mode='binary') # 데이터가 이진 레이블.(개 이거나 고양이라서)

valid_generator = test_datagen.flow_from_directory(
    "C:/Ye_Dong/Tensorflow_Class/P.Song/CNN/CnnData/test", 
    target_size=(150, 150), 
    batch_size=20, 
    class_mode='binary')

# 모델 훈련
history = model.fit(
    train_generator, 
    steps_per_epoch=100, # 20*100 = 총 훈련 데이터 갯수
    epochs=30, 
    validation_data=valid_generator, 
    validation_steps=50)

# 시각화
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) # 이미지를 총 2개만들 것인데, 첫번째에 위치시킴
plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()


plt.subplot(1, 2, 2) 
plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.suptitle('Accuracy & Loss')
plt.tight_layout()

plt.show()