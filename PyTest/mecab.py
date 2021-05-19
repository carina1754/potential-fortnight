import pandas as pd
import numpy as np
import tensorflow as tf
import autokeras as ak
from konlpy.tag import Mecab

train = pd.read_csv("1.csv")
test = pd.read_csv("2.csv")
submission = pd.read_csv("sample_submission.csv")

mecab = Mecab()

train['content'] = train['content'].map(lambda x: ' '.join(mecab.morphs(x)))
test['content'] = test['content'].map(lambda x: ' '.join(mecab.morphs(x)))

x_train = train['content'].values
y_train = train['info'].values

input_node = ak.TextInput()
output_node = ak.TextBlock()(input_node)
output_node = ak.ClassificationHead()(output_node)
clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    overwrite=True,
    max_trials=20)
clf.fit(x_train, y_train, epochs=5)
model = clf.export_model()

tf.keras.backend.clear_session()
model = tf.keras.models.load_model('./auto_model/best_model')

model.fit(x_train, y_train, epochs=100, validation_split=0.2)

pred_test = model.predict(test['title'].values)
pred_test

submission.loc[:, 'level'] = np.where(pred_test > 0.5, 1, 0).reshape(-1)

submission.loc[:, ['title', 'level']].to_csv("v1.0.1.csv", index=False)
submission