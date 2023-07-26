# 필요한 라이브러리
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import matplotlib.pyplot as plt

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