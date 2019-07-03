from sklearn import neighbors
from sklearn import datasets

knn=neighbors.KNeighborsClassifier()
# there is iris(a kind of flower) in the package
iris=datasets.load_iris()
print iris
x=iris.data
print  x
y=iris.target
print y
# build schema
knn.fit(x,y)
# predict kind by 4 features
predictedLabel=knn.predict([[0.1, 0.2, 0.3, 0.4]])
print predictedLabel