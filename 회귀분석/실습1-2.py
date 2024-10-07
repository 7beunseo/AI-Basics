import numpy as np
from sklearn.linear_model import LinearRegression

linear_model= LinearRegression()

X = [[163,0],[162,0],[171,0],[162,0],[164,0],[162,0],[158,0],[173,0], [168,1],[166,1],[173,1],[165,1],[177,1],[163,1],[178,1],[172,1]]
y = [55,51,59,53,61,56,44,57,65,61,68,63,68,61,76,67]
linear_model.fit(X, y)

print(len(X))
print(len(y))

coef = linear_model.coef_
intercept = linear_model.intercept_
score=linear_model.score(X, y)

print ("y = {}*X + {:.2f}".format(coef.round(2), intercept))
print ("데이터와 선형 회귀 직선의 관계점수 :  {:.1%}".format(score))

# import matplotlib.pyplot as plt

# plt.scatter(X, y, color='blue', marker='D')
# y_pred = linear_model.predict(X)
# plt.plot(X, y_pred, 'r:')
# plt.show()

unseen = [[167,1], [167,0]]
result = linear_model.predict(unseen)
print ("키 {}cm는 몸무게 {}kg으로 추정됨".format(unseen, result.round(1)))

