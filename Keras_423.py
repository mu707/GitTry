import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(233)

X = np.linspace(-1, 1, 200)

np.random.shuffle(X)



Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))


plt.show()

X_train, Y_train = X[:160], Y[:160]

X_test, Y_test = X[160:], Y[160:]

#定义model

model = Sequential()   #单输入输出

model.add(Dense(output_dim=1, input_dim=1))


model.compile(loss='mse', optimizer='sgd')

#开始
print("Training----------")
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    #选择 train_on_batch 这个训练函数
    if step % 100==0:
        print("train cost:", cost)

#测试

print("\nTesting--------")
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:',cost)

W, b = model.layers[0].get_weights()

print("Weights:", W, "\nbiases:", b)

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()


