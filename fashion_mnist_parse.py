from fashion_mnist.utils import mnist_reader
import numpy as np
import pickle

X_train, y_train = mnist_reader.load_mnist('fashion_mnist/data/fashion', kind='train')
X_train = X_train.reshape((-1, 28, 28))

print(X_train.shape)
print(y_train.shape)

data_dict = {}
for i in range(10):
    data_dict[str(i)] = X_train[np.where(y_train == i)]
    print(len(data_dict[str(i)]))

with open( "fashion_mnist.pkl", "wb" ) as f:
	pickle.dump(data_dict, f)