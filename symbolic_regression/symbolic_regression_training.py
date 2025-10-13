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
X = np.transpose(log_vars)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

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
    select_k_features=5,
    maxdepth=10,
    batching=True,
    batch_size=1024,
    )
model.fit(
    x_train,
    y_train,
    #variable_names=list(keys)
    )

# saving the model to pickle file
with open(folder+"pysr_model.pkl", "wb") as f:
    pickle.dump(model, f)

mse = np.array(model.equations_.loss)
complexity = np.array(model.equations_.complexity)

print(model)
print(f"Most Accurate - {complexity[-1]}: {model.sympy()}")

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# saving the data to a json file
sr_data = {'keys': keys.tolist(),
           'f_esc_test': y_test.tolist(), 
           'f_esc_train': y_train.tolist(), 
           'f_esc_test_pred': y_test_pred.tolist(), 
           'f_esc_train_pred': y_train_pred.tolist(),
           'test_data': x_test.tolist(),
           'train_data': x_train.tolist(),
           'equation': str(model.sympy())}
with open(folder+'f_esc_sr_test_train.json', 'w') as json_file:
    json.dump(sr_data, json_file)
