import matplotlib.pyplot as plt 
import numpy as np
import os
pathADNI = "C:\\Users\\hassa\\Desktop\\Assignment\\archive\\Alzheimers-ADNI\\"

train_loss = np.load(os.path.join(pathADNI, 'loss_train.npy'))
train_metric = np.load(os.path.join(pathADNI, 'metric_train.npy'))
test_loss = np.load(os.path.join(pathADNI, 'loss_test.npy'))
test_metric = np.load(os.path.join(pathADNI, 'metric_test.npy'))

plt.figure("Results 25 june", (12, 6))
plt.subplot(2, 2, 1)
plt.title("Train dice loss")
x = [i + 1 for i in range(len(train_loss))]
y = train_loss
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.title("Train metric DICE")
x = [i + 1 for i in range(len(train_metric))]
y = train_metric
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 3)
plt.title("Test dice loss")
x = [i + 1 for i in range(len(test_loss))]
y = test_loss
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 4)
plt.title("Test metric DICE")
x = [i + 1 for i in range(len(test_metric))]
y = test_metric
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
