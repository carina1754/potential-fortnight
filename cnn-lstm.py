import pandas as pd
import numpy as np
import tensorflow as tf
import autokeras as ak
from eunjeon import Mecab
from konlpy.tag import Kkma
from konlpy.tag import Okt
import re
train = pd.read_csv('1.csv')
test = pd.read_csv('2.csv')

def text_preprocessing(text_list):
    
    stopwords = ['을', '를', '이', '가', '은', '는', 'null'] #불용어 설정
    tokenizer = Okt() #형태소 분석기 
    token_list = []
    
    for text in text_list:
        txt = re.sub('[^가-힣a-z]', ' ', text) #한글과 영어 소문자만 남기고 다른 글자 모두 제거
        token = tokenizer.morphs(txt) #형태소 분석
        token = [t for t in token if t not in stopwords or type(t) != float] #형태소 분석 결과 중 stopwords에 해당하지 않는 것만 추출
        token_list.append(token)
        
    return token_list, tokenizer

#형태소 분석기를 따로 저장한 이유는 후에 test 데이터 전처리를 진행할 때 이용해야 되기 때문입니다. 
train['new_article'], okt = text_preprocessing(train['title']) 

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length = max_len))

import gensim
word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True)

<<<<<<< HEAD
train = pd.read_csv("1.csv")
change_value_dict = {'판단유보':0,'전혀 사실 아님' : 0,'대체로 사실 아님' : 0.25,'절반의 사실' : 0.5,'대체로 사실' : 0.75, '사실' : 1}
train = train.replace({'level' : change_value_dict})

x_train = train['title']
y_train = train['level']

# 모델의 설정
model = Sequential()
model.add(Embedding(5000, 100))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
  
# 모델의 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
# 모델의 실행
history = model.fit(x_train, y_train, batch_size=100, epochs=5)
  
# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(x_test)[1]))
  
# 테스트셋의 오차
y_vloss = history.history['val_loss']
  
# 학습셋의 오차
y_loss = history.history['loss']
  
# 그래프로 표현
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')
  
# 그래프에 그리드를 주고 레이블을 표시
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
=======

# train['title'] = train['title'].map(lambda x: ' '.join(Mecab.morphs(x)))
# test['title'] = test['title'].map(lambda x: ' '.join(Mecab.morphs(x)))
# x_train = train['title'].values
# y_train = train['level'].values
# # 모델의 설정
# input_node = ak.TextInput()
# output_node = ak.TextBlock()(input_node)
# output_node = ak.ClassificationHead()(output_node)
# clf = ak.AutoModel(
#     inputs=input_node,
#     outputs=output_node,
#     overwrite=True,
#     max_trials=20)
# clf.fit(x_train, y_train, epochs=5)
# model = clf.export_model()

# tf.keras.backend.clear_session()
# model = tf.keras.models.load_model('./auto_model/best_model')


# # 모델의 컴파일
# model.fit(x_train, y_train, epochs=100, validation_split=0.2)
  
# # 모델의 실행
# pred_test = model.predict(test['title'].values)
# print(pred_test)
  
# # 테스트 정확도 출력
# print("\n Test Accuracy: %.4f" % (model.evaluate(x_test)[1]))
  
# # 테스트셋의 오차
# y_vloss = history.history['val_loss']
  
# # 학습셋의 오차
# y_loss = history.history['loss']
  
# # 그래프로 표현
# x_len = numpy.arange(len(y_loss))
# plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
# plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')
  
# # 그래프에 그리드를 주고 레이블을 표시
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
>>>>>>> e1c65ed4ae0c584f9d8617056749315002677425
