import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 필요한 라이브러리

# 데이터 불러오기
dataset_name = 'amazon_us_reviews/Personal_Care_Appliances_v1_00'

data, info = tfds.load(name=dataset_name, split=tfds.Split.TRAIN, with_info=True)

# 데이터 확인
sample_data = list(data.take(5))  # 처음 5개 데이터 확인
for i, example in enumerate(sample_data, 1):
    print(f"Example {i}:")
    print("Review body:", example["data"]['review_body'].numpy().decode('utf-8'))
    print("Star rating:", example["data"]['star_rating'].numpy())
    print()

# 데이터 전처리

# 문장 정제
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string.decode('utf-8')) #영문자, 숫자 외의 문자를 제거하고 'utf-8'형식으로 디코드
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

# 학습 데이터의 리뷰와 레이팅 분리
reviews = [d['review_body'].numpy().decode('utf-8') for d in data]
ratings = np.array([d['star_rating'].numpy() for d in data])

# 데이터 전처리
cleaned_reviews = [clean_str(review) for review in reviews]

# 문장 길이 확인 (토큰화 후 단어 개수)
sentence_lengths = [len(review.split()) for review in cleaned_reviews]

# 문장 길이를 그래프로 시각화
plt.hist(sentence_lengths, bins=50, color='blue', alpha=0.5)
plt.title('Sentence Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()
