import numpy as np
from sklearn.linear_model import LinearRegression

linear_model= LinearRegression()

# 학슴한 데이터
# 왜 리스트 내 리스트 ? -> 기본적으로 2차원 형태로 넣어야함
X = [[163],[179],[166],[169],[171]] # 독립변수 - 키
y = [54, 63, 57, 56, 58] # 알고싶은 타깃 - 몸무게
linear_model.fit(X, y) # g학습 진행

# y = mx + b에서
coef = linear_model.coef_ # 기울기를 구함 = m
intercept = linear_model.intercept_ # b
# 훈련된 데이터의 정답 -> 공부한 거 잘 맞췄냐 못맞췄냐
score=linear_model.score(X, y) # 점수 - 데이터를 넣고 오차를 계산해줌 (오차가 얼마예요?)

print ("y = {}*X + {:.2f}".format(coef.round(2), intercept)) # 소숫점 2번재자리에서 반올림
print ("데이터와 선형 회귀 직선의 관계점수 :  {:.1%}".format(score))

import matplotlib.pyplot as plt # 그래프로 찍어줌

plt.scatter(X, y, color='blue', marker='D')
y_pred = linear_model.predict(X)
plt.plot(X, y_pred, 'r:')
plt.show()
# 점이 데이터
# 만족하는 그래프를 찾음 -> 빨간색 직선

# 키 167일때 결과 어떨 것 같아?
unseen = [[163]]
# 163일때 추정되는 몸무게 값
result = linear_model.predict(unseen) # 새로운 데이터 결과를 줌
print ("키 {}cm는 몸무게 {}kg으로 추정됨".format(unseen, result.round(1)))

