# 필요한 라이브러리
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# 데이터 불러오기
dataset_name = 'amazon_us_reviews/Personal_Care_Appliances_v1_00'
data, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)

# 데이터 크기 확인
num_examples = info.splits['train'].num_examples

# 데이터를 80대 20으로 분할할 인덱스 계산
split_ratio = int(0.8 * num_examples)

# 데이터를 80%를 학습용(train) 데이터로, 나머지 20%를 테스트용(test) 데이터로 분할
train_data = list(data.take(split_ratio))
test_data = list(data.skip(split_ratio))

# 데이터 전처리
import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'{2,}", "\'", string)
    string = re.sub(r"\'", "", string)

    return string.lower()

# 리뷰 데이터와 레이블 분리 (with error handling)
try:
    train_reviews = [clean_str(data['data']['review_body'].numpy().decode("utf-8")) for data in train_data]
    train_ratings = [data['data']['star_rating'] - 1 for data in train_data]  # 레이블 값 1씩 감소

    test_reviews = [clean_str(data['data']['review_body'].numpy().decode("utf-8")) for data in test_data]
    test_ratings = [data['data']['star_rating'] - 1 for data in test_data]  # 레이블 값 1씩 감소
except KeyError:
    print("Error: 'review_body' key not found in the dataset. Check the dataset information or use a different dataset.")
    exit()
# Tokenizer를 사용하여 단어를 숫자로 변환하고 pad_sequences를 사용하여 문장을 패딩
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_reviews)

train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

max_length = 100
train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_length)
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length)

# 별점은 1부터 5까지의 범위를 갖기 때문에, 출력 레이어의 노드 수는 5로 설정
output_nodes = 5

# 간단한 신경망 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=max_length, mask_zero=True),
    tf.keras.layers.LSTM(units=30),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(output_nodes, activation='softmax')  # softmax 활성화 함수 사용
])

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 모델 학습
train_ratings = np.array(train_ratings)
model.fit(train_padded, train_ratings, epochs=10, batch_size=32)

# 테스트 데이터 예측
test_ratings = np.array(test_ratings)
predicted_ratings = model.predict(test_padded)
predicted_ratings = np.argmax(predicted_ratings, axis=1)

# 정확도 계산
accuracy = np.mean(predicted_ratings == test_ratings)

print(f"정확도: {accuracy}")