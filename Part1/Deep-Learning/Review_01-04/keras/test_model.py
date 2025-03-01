from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten

model =Sequential()

model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

# 当标签是1,2,3这样的数字的时候，使用sparse_categorical_crossentropy，
# 当标签是[0,0,0][0,0,1][0,1,0]这样的one-hot编码时使用categorical_crossentropy



model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

