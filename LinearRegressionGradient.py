import numpy as np
from matplotlib.pyplot import plot, show, figure
from sklearn.datasets import load_boston


def norm(x, y, n):
    for i in range(0, n):
        x[:, i] = (x[:, i] - np.mean(x[:, i]))/np.std(x[:, i])
    y = (y - np.mean(y))/np.std(y)

    return x, y


def get_train_validate_test(n, x, y):
    return x[:np.int(n * 0.8)], x[np.int(n * 0.8):np.int(n * 0.9)], x[np.int(n * 0.9):], \
           y[:np.int(n * 0.8)], y[np.int(n * 0.8):np.int(n * 0.9)], y[np.int(n * 0.9):]


def get_train_test(n, x, y):
    return x[:np.int(n * 0.8)], x[np.int(n * 0.8):], \
           y[:np.int(n * 0.8)], y[np.int(n * 0.8):]


def prepare_data(with_valid):
    dataset = load_boston()

    X = dataset.data
    Y = dataset.target

    X, Y = norm(X, Y, 13)

    if with_valid:
        return get_train_validate_test(506, X, Y)
    else:
        return get_train_test(506, X, Y)


def create_mp_matrix(input_x, input_func):
    MP = np.ones((np.size(input_x[:, 0]), 13))

    for num, func in enumerate(input_func):
        MP[:, num] = func(input_x[:, num])

    return MP


def gradient_descent(x_train, y_train, func_cur, lambda_cur, e, learning_rate, iterations_num):
    w = np.random.normal(size=13, scale=0.01)

    mp = create_mp_matrix(x_train, func_cur)
    mp_t = mp.T

    stop_condition = False
    error = []

    for i in range(1000):
        E = w @ (mp_t @ mp + lambda_cur * np.identity(13)) - y_train.T @ mp

        if np.linalg.norm(E) < e or np.linalg.norm(learning_rate * E) < e:
            stop_condition = True
        else:
            w -= learning_rate * E
            error.append((np.sum((mp @ w.T - y_train) ** 2) + lambda_cur * w @ w.T) / 2)
    return w, error


def get_error(x, y, w, lambda_use, func):
    mp = create_mp_matrix(x, func)
    return (np.sum((mp @ w.T - y) ** 2) + lambda_use * w @ w.T) / 2


def graph(error, color):
    x = np.arange(np.size(error))

    figure()
    plot(error, color=color)
    show()


def train():
    x_train, x_test, y_train, y_test, = prepare_data(with_valid=False)

    lambda_cur = 0.0001
    func_set = [lambda x: x]
    learning_rate = 0.000001
    epsilon = 0.0001

    iterations_num = 10000

    func_cur = np.random.choice(func_set, 13)

    w_cur, errors = gradient_descent(x_train, y_train, func_cur, lambda_cur, epsilon, learning_rate, iterations_num)

    train_err = get_error(x_train, y_train, w_cur, lambda_cur, func_cur)
    test_err = get_error(x_test, y_test, w_cur, lambda_cur, func_cur)

    graph(errors, 'red')

    print("Train error: ", train_err)
    print("Test error: ", test_err)


def train_with_validate():
    x_train, x_test, x_validate, y_train, y_test, y_validate = prepare_data(with_valid=True)

    lambda_set = [0, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]
    func_set = [lambda x: x ** 9, lambda x: x ** 8, lambda x: x ** 7, lambda x: x ** 6, lambda x: x ** 5,
                lambda x: x ** 4, lambda x: x ** 3, lambda x: x ** 2, lambda x: x, lambda x: 1]

    learning_rate_set = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    e_set = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]

    error_best = np.inf
    w_best = None
    lambda_best = None
    func_best = None
    train_errors_best = None
    learning_rate_best = None
    e_best = None

    iterations_num = 20000

    for i in range(iterations_num):

        print(i)

        lambda_cur = np.random.choice(lambda_set)
        func_cur = np.random.choice(func_set, 13)
        learning_rate = np.random.choice(learning_rate_set)
        e = np.random.choice(e_set)

        w_cur, errors = gradient_descent(x_train, y_train, func_cur, lambda_cur, e, learning_rate, iterations_num)
        e_cur = get_error(x_validate, y_validate, w_cur, lambda_cur, func_cur)

        if e_cur < error_best:
            w_best = w_cur
            error_best = e_cur
            lambda_best = lambda_cur
            func_best = func_cur
            train_errors_best = errors
            e_best = e
            learning_rate_best = learning_rate

    train_err = get_error(x_train, y_train, w_best, lambda_best, func_best)
    test_err = get_error(x_test, y_test, w_best, lambda_best, func_best)

    graph(train_errors_best, 'blue')

    print("Train error: ", train_err)
    print("Test error: ", test_err)
    print("Used lambda: ", lambda_best)
    print("Used epsilon: ", e_best)
    print("Used learning rate: ", learning_rate_best)
    print("Used functions: ", func_best)


#train_with_validate()

train()
