# -*- coding: utf-8 -*-

from BO.bayesian_optimization import BayesianOptimization

import sys
from collections import OrderedDict
import numpy as np
import tensorflow as tf

# data mnist
from mnist_data.mnist import download_mnist, load_mnist, key_file
download_mnist()
X_train = load_mnist(key_file["train_img"])[8:, :]
X_test = load_mnist(key_file["test_img"], )[8:,:]
y_train = load_mnist(key_file["train_label"], 1)
y_test = load_mnist(key_file["test_label"], 1)


# 目的関数（ハイパーパラメータを引数にする関数）
def MLP(alpha, lr, layer1, layer2, layer3):
    X = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.int32, [None, ])
    y_ = tf.one_hot(label, depth=10, dtype=tf.float32)

    w_0 = tf.Variable(tf.random_normal([784, int(layer1)], mean=0.0, stddev=0.05))
    b_0 = tf.Variable(tf.zeros([int(layer1)]))
    h_0 = tf.sigmoid(tf.matmul(X, w_0) + b_0)

    w_1 = tf.Variable(tf.random_normal([int(layer1), int(layer2)], mean=0.0, stddev=0.05))
    b_1 = tf.Variable(tf.zeros([int(layer2)]))
    h_1 = tf.sigmoid(tf.matmul(h_0, w_1) + b_1)

    w_2 = tf.Variable(tf.random_normal([int(layer2), int(layer3)], mean=0.0, stddev=0.05))
    b_2 = tf.Variable(tf.zeros([int(layer3)]))
    h_2 = tf.sigmoid(tf.matmul(h_1, w_2) + b_2)

    w_o = tf.Variable(tf.random_normal([int(layer3), 10], mean=0.0, stddev=0.05))
    b_o = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h_2, w_o) + b_o)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    L2_sqr = tf.nn.l2_loss(w_0) + tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2)

    loss = cross_entropy + alpha * L2_sqr
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print("Training...")
        for i in range(20000):
            #batch_x, batch_y = X_train[(50*i):(50*(i+1)),:], y_train[(50*i):(50*(i+1)),0]
            batch_index = np.random.choice(X_train.shape[0], 50, replace=False)
            batch_x = X_train[batch_index, :]
            batch_y = y_train[batch_index, 0]

            train_step.run({X: batch_x, label: batch_y})
            if i % 2000==0:
                train_accuracy = accuracy.eval({X: batch_x, label: batch_y})
                #print(" %6d %6.3f" % (i, train_accuracy))
        accuracy = accuracy.eval({X: X_test, label: y_test[:,0]})
        print("accuracy %6.3f" % accuracy)
        return accuracy

def main(k_num, acq, verbose=True):
    gp_params = {"alpha": 1e-5}

    # ハイパーパラメータの範囲を指定
    BO = BayesianOptimization(MLP,
                              {"alpha": (1e-8, 1e-4), "lr": (1e-6, 1e-2),
                               "layer1": (10, 100),"layer2": (10, 100),"layer3": (10, 100)},
                              verbose=verbose, kernel_num = k_num)

    BO.explore({"alpha": [1e-8, 1e-8, 1e-4, 1e-4],"lr": [1e-6, 1e-2, 1e-6, 1e-2],
                "layer1": [10, 50, 100, 50], "layer2": [10, 50, 100, 50],"layer3": [10, 50, 100, 50]})

    BO.maximize(n_iter=200, acq=acq, **gp_params)

    print("-"*53)
    print("Final Results")
    print("kernel: {}".format(str(BO.kernel)))
    print("acquisition function: {}".format(BO.acquisition))

    print("score: {}".format(BO.res["max"]["max_val"]))
    print("best_parameter: ")
    print(BO.res["max"]["max_params"])
    print("-"*53)

if __name__ == "__main__":
    main(0, "ucb")
    # kernel function
    #   0: Matern(nu=0.5)
    #   1: Matern(nu=1.5)
    #   2: Matern(nu=2.5)
    #   3: RBF
    #   bayesian_optimization.pyにて追加、変更、可能
    # acquisition function
    #   ucb
    #   ei
    #   poi
    #   helpers.pyにて追加可能
