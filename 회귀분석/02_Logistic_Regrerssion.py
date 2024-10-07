from sklearn.linear_model import LogisticRegression

# 제공된 데이터
X = [[168, 0],[166, 0],[173, 0],[165, 0],[177, 0], [163, 61], [178, 0], [172, 0], [163, 1], [162, 1], [171, 59], [162,1], [164, 1],[162,1], [158, 1], [173, 1]] # 독립변수 - 키
y = [65, 61, 68, 63, 68, 61, 76, 67, 55, 51, 59, 53, 61, 56, 44, 57] # 알고싶은 타깃 - 몸무게

# 60이 넘어가면 1이다. 그렇지 않으면 0
y_binary = [1 if weight >= 60 else 0 for weight in y]

# 로지스틱 회귀
logistic_model = LogisticRegression()
# fit : 학습하겠다.
# Linear에서는 x와 y가 그대로 들어갔지만, Y_binary값이 들어가야 함 (101010 으로 구성된 배열)
logistic_model.fit(X, y_binary)

# 계수와 절편, 점수
print('계수:', logistic_model.coef_)
print('절편:', logistic_model.intercept_)
# 학습한 데이터르 기반으로 점수를 매김
print('점수:', logistic_model.score(X, y_binary))

testX = [ [167, 0], [167, 1] ]
# 예측 확률
y_pred = logistic_model.predict_proba(testX)
print('예측 확률:', y_pred)
# 확률값이 4개 나옴 -> 가로로 2개를 더하면 1
# [0.41518704 0.58481296] -> 더하면 1
# 몸무게를 공부함 -> 60kg가 넘어가면 결과가 1, 60kg 안 넘어가면 0일 것
# 첫번째 = 60kg 넘어가지 않을 확률, 두번째 확률 60kg 넘어갈 확률

# [0.41666545 0.58333455] -> 더하면 1
# 첫번째 = 60kg 넘어가지 않을 확률, 두번째 확률 60kg 넘어갈 확률
# 60kg가 넘을 확률이 더 크기 대문에 1일 출력되고, 반대라면 0이 출력됨

# 예측 결과
y_pred_logistic = logistic_model.predict(testX)
print('예측 결과:', y_pred_logistic)
