"""
This code runs symbolic regression on a test dataset to find an equation for f_esc.
"""

from sr_functions import prepare_data_sr
import numpy as np
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import matplotlib.pyplot as plt

folder = "symbolic_regression/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 0
# True if model is generated to predict for an observational catalogue 
obvs = False

keys, log_vars, Y = prepare_data_sr(file, f_or_n, obvs)
print(keys)
# nan_indices = [index for index, val in enumerate(Y) if val <=-3 ][::-1]
# print(f"rows deleted: {len(nan_indices)}")
# Y = np.delete(Y, nan_indices)
# log_vars = np.delete(log_vars, nan_indices, axis=1)
X = log_vars
X[5] = (-X[6] - 27.822582)/(-X[1] - X[3] + 6.5152617)
X = np.transpose(X[5:7])
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# training the symbolic regresion model
model = PySRRegressor(
    populations=50,
    niterations=500,
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["pow10(x) = 10 ^ x"],
    # unary_operators=["square", "inv", "sqrt", "exp", "log", "abs"],
    extra_sympy_mappings={"pow10": lambda x: 10**x},
    model_selection="accuracy",
    # select_k_features=5,
    maxdepth=10,
    batching=True,
    batch_size=1024,
    )
model.fit(
    x_train,
    y_train,
    #variable_names=list(keys)
    )

mse = np.array(model.equations_.loss)[::-1]
complexity = np.array(model.equations_.complexity)
knee_index = -1
for index in range(len(mse)):
    try:
        if mse[index+1] - mse[index] > 0.01:
            knee_index = len(mse) - (index+1)
            break
    except:
        break
mse = mse[::-1]

print(model)
print(f"Most Accurate - {complexity[-1]}: {model.sympy()}")
print(f"Optimal - {complexity[knee_index]} : {model.sympy(knee_index)}")

y_train_pred = model.predict(x_train, knee_index)
y_test_pred = model.predict(x_test, knee_index)

# saving the model to pickle file data to a json file
with open(folder+"pysr_model.pkl", "wb") as f:
    pickle.dump(model, f)
sr_data = {'keys': keys.tolist(),
           'f_esc_test': y_test.tolist(), 
           'f_esc_train': y_train.tolist(), 
           'f_esc_test_pred': y_test_pred.tolist(), 
           'f_esc_train_pred': y_train_pred.tolist(),
           'test_data': x_test.tolist(),
           'train_data': x_train.tolist(),
           'equation': str(model.sympy(knee_index))}
with open(folder+'f_esc_sr_test_train.json', 'w') as json_file:
    json.dump(sr_data, json_file)

plt.plot(complexity, mse, marker='o')
plt.axvline(complexity[knee_index], color='red', linestyle='dashdot')
plt.show()
