
import numpy as np
np.random.seed(233)

from keras.datasets import mnist
from  keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

#下载数据
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

#预处理
X_train = X_train.reshape(X_train.shape[0], -1)/255
X_test = X_test.reshape(X_test.shape[0], -1)/255
Y_train = np_utils.to_categorical(Y_train,num_classes=10)
Y_test = np_utils.to_categorical(Y_test,num_classes=10)

#构建网络
model = Sequential([
    Dense(100, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation("softmax"),
])

#定义优化器
rmsprop = RMSprop(lr=0.01, rho=0.9, epilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#使用fit训练网络

print("Training______________")
model.fit(X_train, Y_train, epochs=4, batch_size=32)

print("\nTesting______________")
#评价
loss, accuracy = model.evaluate(X_test, Y_test)

print("test loss:", loss)
print("tset accuracy:", accuracy)







