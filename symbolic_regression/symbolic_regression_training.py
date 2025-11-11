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
import multiprocessing

folder = "symbolic_regression/"
file = 'cat.hdf5'

# 0 for f_esc, 1 for n_esc
f_or_n = 0
# True if model is generated to predict for an observational catalogue 
obvs = True

keys, log_vars, Y, resolution = prepare_data_sr(file, f_or_n=f_or_n, basic=False, obvs=obvs)
print(keys)
# nan_indices = [index for index, val in enumerate(Y) if val <=-3 ][::-1]
# print(f"rows deleted: {len(nan_indices)}")
# Y = np.delete(Y, nan_indices)
# log_vars = np.delete(log_vars, nan_indices, axis=1)
features = ([2, 3, 4], [0, 1, 3])[f_or_n]
X = np.transpose(log_vars[features])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

x_train, x_test, y_train, y_test, res_train, res_test = train_test_split(
    X, Y, resolution, test_size=0.25, random_state=0)

# training the symbolic regresion model
n_cores = multiprocessing.cpu_count()
model = PySRRegressor(
    populations=100,
    population_size=100,
    niterations=100,
    binary_operators=["*", "+", "-", "/"],
    # unary_operators=["pow10(x) = 10 ^ x"],
    unary_operators=["exp", "log"],
    # extra_sympy_mappings={"pow10": lambda x: 10**x},
    model_selection="accuracy",
    # select_k_features=5,
    maxdepth=5,
    batching=True,
    batch_size=4096,
    procs=n_cores,
    parallelism='multiprocessing',
    turbo=True,
    # parsimony punishes complexity in the score function
    # parsimony=0.001,
    # adaptive_parsimony_scaling=1000,
    # warmup_maxsize_by=0.25,
    # use_frequency=True,
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

f_or_n_str = ('f_esc', 'n_esc')[f_or_n]
# saving the data to a json file
sr_data = {'keys': keys.tolist(),
           'esc_test': y_test.tolist(), 
           'esc_train': y_train.tolist(), 
           'esc_test_pred': y_test_pred.tolist(), 
           'esc_train_pred': y_train_pred.tolist(),
           'test_data': x_test.tolist(),
           'train_data': x_train.tolist(),
           'equation': str(model.sympy()),
           'res_train': res_train.tolist(),
           'res_test': res_test.tolist()}
with open(folder + f_or_n_str + '_sr_test_train.json', 'w') as json_file:
    json.dump(sr_data, json_file)
