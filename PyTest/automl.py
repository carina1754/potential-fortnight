import pandas as pd
import numpy as np
import tensorflow as tf
import autokeras as ak

from konlpy.tag import Mecab

mecab = Mecab()

train = pd.read_csv("1.csv")
test = pd.read_csv("2.csv")
# # change_value_dict = {'판단유보':0,'전혀 사실 아님' : 0,'대체로 사실 아님' : 0.25,'절반의 사실' : 0.5,'대체로 사실' : 0.75, '사실' : 1}
# train = train.replace({'level' : change_value_dict})
print(train)
train['title'] = train['title'].map(lambda x: ' '.join(mecab.morphs(x)))
test['title'] = test['title'].map(lambda x: ' '.join(mecab.morphs(x)))
print 
X = train['title'].values
Y =  train['level'].values

input_node = ak.TextInput()
output_node = ak.TextToIntSequence()(input_node)
output_node = ak.Embedding()(output_node)
output_node = ak.ConvBlock(separable=True)(output_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=5
    )
clf.fit(X, Y, epochs=5)
model = clf.export_model()

test = pd.read_csv("2.csv")
sample_submission = pd.read_csv("sample_submission.csv")
pred_test = model.predict(test['title'].values)
sample_submission.loc[:,'level']=np.where(pred_test)
sample_submission.loc[:,["title","level"]].to_csv("sample_submission.csv", index = False)