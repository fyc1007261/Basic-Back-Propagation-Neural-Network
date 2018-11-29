import numpy as np


# 5000 sets of data, 4 dimensional
left = np.random.rand(5000, 4)
label = left[:, 0] + 5*left[:, 1] + left[:, 2] - left[:, 3]
label = label
print(left.shape, label.shape)


f1 = open('data/data.csv', 'w')
f2 = open('data/label.csv', 'w')
for i in range(left.shape[0]):
    line = str(left[i, :].tolist())[1:-1]
    f1.write(line + '\n')
    f2.write(str(label[i]) + '\n')

f1.close()
f2.close()