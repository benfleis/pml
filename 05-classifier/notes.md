# 05 classifier (discering)

## logistic regression
- logistic function aka sigmoid: = 1 / (1 + e^-z)
- produces range:[0, 1]
- smooth continuous, needed for gradient descent

## updates: loss
- loss func no longer smooth/concave, many local minima due to sigmoid over stepped output
- use _log loss_ instead, achieves same properties and allows similar/same (re)use of train/loss mechanisms
