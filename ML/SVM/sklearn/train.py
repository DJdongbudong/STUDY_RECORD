"""
=====================================================================
SVM Example made in DJH
=====================================================================
fuction:
	input:	
		(x,y),...(x,y);
		label,...labeln;
	output:	

=====================================================================
data:20190531
=====================================================================
"""
print(__doc__)

"""
# data and label======================================================
"""
'''
from sklearn import datasets
iris = datasets.load_iris()
# data: 150 [[pointx,pointy]] 150 points
X = iris.data[:, :2]
# label:150 label
y = iris.target
print(">>> X: <<<\n", X)
print(">>> y: <<<\n", y) 
'''
"""
# our data============================================================
"""
import numpy as np
X = np.array([[5.1,3.5], [4.9, 3.0], [4.7, 3.2], [4.6, 3.1], [1.0, 1.6], [1.4, 1.9], [1.6, 1.4], [1.0, 1.4], [1.4, 1.9], [1.9,1.1]])
y = np.array([2, 2, 0, 0, 1, 1, 1, 1, 1, 1])
print(">>> X: <<<\n", X)
print(">>> y: <<<\n", y) 

"""
# train===============================================================
"""
from sklearn import svm
rbf_svc = (svm.SVC(kernel='rbf', gamma=.5).fit(X, y), y)
print(">>> rbf_svc: <<<\n", rbf_svc)
print("X[:, 0]", X[:, 0])


"""
# drawing============================================================
"""
# mash the map:
import numpy as np
h = .02
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# source the data :
import matplotlib.pyplot as plt
color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
print(">>> list: <<<\n", list(enumerate(rbf_svc)))
for (i, j) in enumerate((rbf_svc)):

	if i == 0:
		clf = j
		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		# Put the result into a color plot
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
		plt.axis('off')
		pass

	if i == 1:
		y_train = j
		# Plot also the training points
		colors = [color_map[y] for y in y_train]
		#绘图,参数s:点的大小，marker:点的形状 alpha:点的亮度，label:标签
		plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='black')
		plt.title('SVC with rbf kernel')
		pass
plt.show()


"""
# save and load ======================================================
"""
from sklearn.externals import joblib
joblib.dump(clf, 'model.pkl')
clf = joblib.load('model.pkl')

"""
# predict=============================================================
"""
result = clf.predict([[-0.8, -1]])
print("predict:", result)


