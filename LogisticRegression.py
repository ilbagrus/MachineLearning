from sklearn.datasets import load_digits
from matplotlib.pyplot import plot, show, figure
import numpy as np


def norm(inp):
    for i in range(0, len(inp[0, :])):
        std = np.std(inp[:, i])
        if std != 0:
            inp[:, i] = (inp[:, i] - np.mean(inp[:, i])) / std
        else:
            inp[:, i] = 0

    return inp


def split_data(inp, res):

    temp = np.column_stack((res, inp))
    np.random.shuffle(temp)

    N = temp[:, 0].size

    return temp[:np.int(N * 0.85), 1:], temp[:np.int(N * 0.85), 0], \
           temp[np.int(N * 0.85):, 1:], temp[np.int(N * 0.85):, 0]


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_accuracy():
    pred_train = np.argmax(vectorU(train), axis=1)
    pred_validate = np.argmax(vectorU(validate), axis=1)
    target_train = np.argmax(train_encoded_targets, axis=1)
    target_valid = np.argmax(validate_encoded_targets, axis=1)

    bool_train = (pred_train == target_train)
    bool_validate = (pred_validate == target_valid)

    return np.mean(bool_train), np.mean(bool_validate)


def logistic_regression(w, b, iterations_num):
    cur_iteration = 0

    saved_E = []
    saved_accuracy_train = []
    saved_accuracy_validate = []
    saved_index = []

    while True:
        Y = vectorSoftmax(vectorU(train))

        cur_E = np.sum(-vectorE(train_encoded_targets.T, np.log(Y.T))) + lambda_cur*np.sum(w**2)/2

        w -= learning_rate * (Y - train_encoded_targets).T @ train + lambda_cur * w
        b -= learning_rate * (Y - train_encoded_targets).T @ np.ones(Y[:, 0].size)

        cur_accuracy_train, cur_accuracy_validate = get_accuracy()

        saved_E.append(cur_E)
        saved_accuracy_train.append(cur_accuracy_train)
        saved_accuracy_validate.append(cur_accuracy_validate)
        saved_index.append(cur_iteration)

        cur_iteration += 1

        if cur_E < e or learning_rate * cur_E < e:
            break
        if cur_iteration > iterations_num:
            break
        if cur_iteration % 5 == 0:
            print(cur_E, cur_accuracy_train, cur_accuracy_validate)
    return saved_index, saved_accuracy_validate, saved_accuracy_train, saved_E


def calc_u(x):
    return x@W.T + B


def calc_e(t, y):
    return np.sum(t*y)


def softmax(u):
    u_x = u - np.max(u)
    e_x = np.exp(u_x)
    out = e_x / e_x.sum()

    return out


def graph(x, y, color):
    figure()
    plot(x, y, color=color)
    show()


digits = load_digits()

train, train_res, validate, validate_res = split_data(norm(digits.data), digits.target)

train_encoded_targets = get_one_hot(train_res.T.astype(int), 10)
validate_encoded_targets = get_one_hot(validate_res.T.astype(int), 10)

learning_rate_set = [0.0001, 0.001, 0.01]
lambda_set = [0.0005, 0.0001, 0.001, 0.01]
sigma_set = [0.01, 0.05, 0.1]
init_random_set = [lambda x: 2 * sigma * np.random.random(x) - sigma,
                   lambda x: np.random.normal(size=x, scale=sigma),
                   lambda x: np.random.normal(size=(x[0], x[1]) if type(x) is tuple else x,         # Xavier
                                              scale=np.sqrt(1/(x[0]*x[1] if type(x) is tuple else x))),
                   lambda x: np.random.normal(size=(x[0], x[1]) if type(x) is tuple else x,         # He
                                              scale=2/(x[0]*x[1]) if type(x) is tuple else 2/x)]

lambda_cur = np.random.choice(lambda_set)
learning_rate = np.random.choice(learning_rate_set)
sigma = np.random.choice(sigma_set)
init_func = np.random.choice(init_random_set)
e = 0.001

iter_num = 100

W = init_func((10, train[0, :].size))
B = init_func(10)

vectorSoftmax = np.vectorize(softmax, signature='(a)->(b)')
vectorU = np.vectorize(calc_u, signature='(a)->(b)')
vectorE = np.vectorize(calc_e, signature='(a),(b)->()')


index, accuracy_validate, accuracy_train, E = logistic_regression(W, B, iter_num)

graph(index, E, 'red')
graph(index, accuracy_train, 'green')
graph(index, accuracy_validate, 'blue')
print("Final accuracy: ", accuracy_validate[-1])
