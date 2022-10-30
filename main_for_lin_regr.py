# author: Olga Okrut. Email: vokrut42sv@gmail.com

import numpy as np
import sklearn.datasets as dt
from linear.annealing_lin_regression import QALinearRegression
from utils import get_data_for_training
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    # small test on artificial data. Change the precision vector before running.

    # # define a random dataset for a qa linear regression to test
    # seed = 11
    # rand_state = 11
    # rand= np.random.RandomState(seed)   
    # x_points, y_labels = dt.make_regression(n_samples=10, n_features=1, noise=1, random_state=rand_state)

    # # define a qa linear regression
    # qa_lin_model = QALinearRegression()
    # qa_lin_model.train(x_points, y_labels)
    # y_predicted = qa_lin_model.predict(x_points)
    # print("y_predicted = ", y_predicted)

    # model = LinearRegression()
    # trained_model = model.fit(x_points, y_labels)
    # y_predict_classical = trained_model.predict(x_points)
    # # # plot and save file
    # plt.scatter(x_points, y_labels)
    # plt.plot(x_points, y_predict_classical, color='pink', linewidth=3, label='Classical')
    # plt.plot(x_points, y_predicted, color='green', linewidth=1, label='Quantum')
    # plt.legend(loc='upper left')
    # plt.title("Adiabatic vs Classical Linear Regression on Artificial Data")
    # plt.savefig("results/artificial.png")
    # plt.show()


    age, qrs = get_data_for_training("QRSAxis") #get data

    # define and train an adiabatic Linear regresson.
    print("Initialize and train Quantum Linear Regression . . .")
    qrs_lin_regression = QALinearRegression()
    qrs_lin_regression.train(age.reshape(-1, 1), qrs)

    age_test = np.array([26]).reshape(-1, 1)
    qrs_predicted = qrs_lin_regression.predict(age.reshape(-1,1))

    # print('age_test = ', age_test)
    # print("qrs_predicted by quantum = ", qrs_predicted)

    # compare to a classical linear regression in Scikit-learn
    print("Initialize and train Classical Linear Regression . . .")
    model= LinearRegression()
    trained_model = model.fit(age.reshape(-1, 1), qrs)
    y_predict = trained_model.predict(age.reshape(-1,1))
    # print("predicted by linear = ", y_predict)
    # print("model coeff = ", model.coef_)

    # plat and save file
    plt.scatter(age, qrs)
    plt.plot(age, y_predict, color='pink', linewidth=3, label='Classical')
    plt.plot(age, qrs_predicted, color='green', linewidth=1, label='Quantum')
    plt.legend(loc='upper left')
    plt.title("QRS Axis (in degrees) Depending on Age")
    plt.savefig("results/quantum.png")
    plt.show()

    # model verification.
    print("Evaluate Quantum Linear Regression . . .")
    age_verif = np.array([58, 76, 79, 57, 52, 56, 73, 50, 33, 68, 70, 53])
    qrs_axis_actual = np.array([-20, 34, 42, 45, 30, 16, 12, 30, 35, 53, 55, 2])

    qrs_predicted = qrs_lin_regression.predict(age_verif.reshape(-1,1))

    # print metrics
    print('MAE = ', qrs_lin_regression.mae(qrs_axis_actual, qrs_predicted))
    print('MSE = ', qrs_lin_regression.mse(qrs_axis_actual, qrs_predicted))
    print('Root MSE = ', qrs_lin_regression.root_mse(qrs_axis_actual, qrs_predicted))
    print('R^2 = ', qrs_lin_regression.r_score(qrs_axis_actual, qrs_predicted))