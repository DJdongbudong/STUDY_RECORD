# 输入
数据类型
data = [x,y], ..., [x,y];
label = [0, ..., n]
```
import numpy as np
X = np.array([[5.1,3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [1.0, 1.6], [1.4, 1.9], [1.6, 1.4], [1.0, 1.4], [1.4, 1.9], [1.9,1.1]])
y = np.array([2, 2, 0, 0, 1, 1, 1, 1, 1, 1])
```
# 训练并画图：
```
python train.py
```
![image](https://github.com/DJdongbudong/STUDY_RECORD/blob/master/ML/SVM/sklearn/svm_class2_sklearn.png)
# 预测：
```
python predict.py
```

