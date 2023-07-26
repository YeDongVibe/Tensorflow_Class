import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 데이터셋 로드
data, info = tfds.load(name='amazon_us_reviews/Personal_Care_Appliances_v1_00', split=tfds.Split.TRAIN, with_info=True)

# 데이터 전처리
reviews = [example['data']['review_body'].numpy().decode('utf-8') for example in data]
labels = [example['data']['star_rating'].numpy() for example in data]
labels = np.array(labels) > 3  # Positive일 경우 True, Negative일 경우 False로 변환

# 정규표현식을 이용하여 띄어쓰기 기준으로 분리하여 보여주기
print("띄어쓰기 기준으로 분리된 토큰들:")
for review in reviews[:5]:
    tokens = re.findall(r'\b\w+\b', review)
    print(tokens)

# 각 문장의 단어 길이 확인 (y축은 문장의 단어 수, x축은 문장 수로 그래프를 그림)

# 단어 정제 및 각 단어의 최대 길이를 조정

# Tokenizer와 pad_sequences를 사용한 문장 전처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
max_sequence_length = max(len(sequence) for sequence in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Tokenizer의 동작 확인 - texts_to_sequences()이 하는 일을 실행으로 확인 (공백은 0으로 변환)

# 감성 분석을 위한 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 감성 분석 모델 학습
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("감성 분석 모델 학습 중...")
print(model.summary())
model.fit(padded_sequences, labels, epochs=5, batch_size=32)

# 감성 분석 모델 학습 결과 확인
# 여기에는 학습 결과에 관련된 내용이 들어갑니다.

# 테스트 데이터 평가
print("테스트 데이터 평가 중...")
loss, accuracy = model.evaluate(padded_sequences, labels)
print("테스트 데이터 정확도:", accuracy)

# 임의의 문장 감성 분석 결과 확인
test_sentences = ["This product is amazing!",
                  "I don't like this at all.",
                  "It works well for me.",
                  "Not satisfied with the quality."]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')
print("임의의 문장 감성 분석 결과:")
print(model.predict(test_padded_sequences))
