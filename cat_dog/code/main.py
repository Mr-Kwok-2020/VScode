import h5py
import numpy as np
from sklearn.utils import shuffle

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())




X_train = []
X_test = []

for filename in ["gap_my_ResNet50.h5", "gap_my_Xception.h5", "gap_my_InceptionV3.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

X_train, y_train = shuffle(X_train, y_train)





from keras.models import *
from keras.layers import *

input_tensor = Input(X_train.shape[1:])
x = input_tensor
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=256, epochs=500, validation_split=0.2)
model.save('model_my.h5')
y_pred = model.predict(X_test, verbose=1)
y_pred = y_pred.clip(min=0.005, max=0.995)
import pandas as pd
from keras.preprocessing.image import *

df = pd.read_csv("pred.csv")
df2 = pd.read_csv("pred_name.csv")
image_size = (224, 224)
gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("D:\\data\\cat_dog\\test", image_size, shuffle=False,
                                         batch_size=16, class_mode=None)

for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('\\')+1:fname.rfind('.')])
    df.at[index-1, 'label'] = y_pred[i]
    if y_pred[i]>0.50:
        df2.at[index-1, 'label'] = "狗"
    else:
        df2.at[index-1, 'label'] = "猫"


df.to_csv('pred.csv', index=None)
df.head(10)
df2.to_csv('pred_name.csv', index=None)
df2.head(10)


















