import numpy as np
from matplotlib.pyplot import plot, show, figure


def create_pm(input_x, input_func):
    PM = np.ones((np.size(input_x), np.size(input_func)))

    for num, func in enumerate(input_func):
        PM[:, num] = func(input_x)
    return PM


def train():
    PM = create_pm(train_data[:, 0], func_cur)
    PM_T = PM.T

    return np.linalg.inv(PM_T @ PM + lambda_cur * np.identity(np.size(func_cur))) @ PM_T @ train_data[:, 1]


def validate(data, w, func, lambda_use):
    model = create_pm(data[:, 0], func) @ w

    e = (data[:, 1] - model) ** 2

    E = (np.sum(e) + lambda_use * (w @ w.T))
    return E/2


def split_data():
    temp = np.column_stack((x, z))
    np.random.shuffle(temp)
    return np.sort(temp[:np.int(N*0.8)], axis=0), np.sort(temp[np.int(N*0.8):np.int(N*0.9)], axis=0), np.sort(temp[np.int(N*0.9):], axis=0)


def graph(data, w, func):
    model = create_pm(data[:, 0], func) @ w

    figure()
    plot(x, t, color='blue', linestyle=' ', marker='o', markersize='0.1')
    plot(x, z, color='green')
    plot(data[:, 0], model, color='red')
    show()


N = 10000
x = np.linspace(0, 1, N)

z = 20*np.sin(2 * np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

train_data, valid_data, test_data = split_data()

func_set = [np.sin, np.cos, np.exp, np.sqrt, lambda sn6: np.sin(6 * np.pi * sn6),
             lambda x8: x8**8, lambda x7: x7**7, lambda x6: x6**6, lambda x5: x5**5, lambda x4:x4**4, lambda x3:x3**3,
             lambda x2: x2**2, lambda x1:x1, lambda x0: 1]

lambda_set = [0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

e_best = np.inf
w_best = None
lambda_best = None
func_best = None

iterations_num = 1000

for i in range(iterations_num):
    lambda_cur = np.random.choice(lambda_set)
    func_cur = np.random.choice(func_set, np.random.randint(1, np.size(func_set) + 1), replace=False)

    w_cur = train()
    e_cur = validate(valid_data, w_cur, func_cur, lambda_cur)

    if e_cur < e_best:
        w_best = w_cur
        e_best = e_cur
        lambda_best = lambda_cur
        func_best = func_cur

graph(test_data, w_best, func_best)
err = validate(test_data, w_best, func_best, lambda_best)

print(lambda_best, func_best, err)
