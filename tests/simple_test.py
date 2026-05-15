# tests/simple_test.py
from bqdp import BQDP





model = BQDP(n=30)
candidates = 15
history = [
    3,
    7,
    12,
    4,
    7,
    2,
    1,
    1,
    5,
    # 6, # This will be the prediction (int_to_predict)
]
int_to_predict = 6

for correct in history:
    model.update(correct)

candidates = model.predict(k=candidates)
print(candidates)

if int_to_predict in candidates:
    print("Correct!")
    model.update(correct=int_to_predict)
else:
    print("Wrong!")