# author: Olga Okrut. GitHub: olgOk

import numpy as np
import sklearn.datasets as dt
from linear.annealing_lin_regression import QALinearRegression
from utils import get_data_for_training
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

if __name__ == "__main__":

    age, qrs = get_data_for_training("QRSAxis") #get data

    age_train, age_test, qrs_train, qrs_test = train_test_split(age, qrs, test_size=0.25, random_state=42) 
    # define and train an adiabatic Linear regresson.
    print("Initialize and train Quantum Linear Regression . . .")
    precision_vector = np.array([0, 0.125, 0.25, 0.5, 0.75])
    qrs_lin_regression = QALinearRegression(precision_vector=precision_vector, normalize=True, scaler="RobustScaler")
    qrs_lin_regression.train(age_train.reshape(-1, 1), qrs_train)
    qrs_predicted = qrs_lin_regression.predict(age_train.reshape(-1,1))

    # compare to a classical linear regression in Scikit-learn
    print("Initialize and train Classical Linear Regression . . .")
    model= LinearRegression()
    trained_model = model.fit(age_train.reshape(-1, 1), qrs_train)
    y_predict = trained_model.predict(age_train.reshape(-1,1))

    print("R^2 on training dataset for Scikit-Learn:", r2_score(qrs_train, y_predict))
    print("R^2 on training dataset for Adiabatic Linear Regression:", qrs_lin_regression.r_score(qrs_train, qrs_predicted))
    # plot and save file
    plt.scatter(age_train, qrs_train)
    plt.plot(age_train, y_predict, color='pink', linewidth=2, label='Classical')
    plt.plot(age_train, qrs_predicted, color='green', linewidth=3, label='Quantum')
    plt.legend(loc='upper left')
    plt.title("QRS Axis (in degrees) Depending on Age")
    plt.savefig("results/sklearn_vs_adiabatic_train.png")
    plt.show()

    # model evaluation.
    print("Evaluate Quantum Linear Regression . . .")
    qrs_predicted_eval = qrs_lin_regression.predict(age_test.reshape(-1,1))
    qrs_classical = model.predict(age_test.reshape(-1,1))

    print("Age {}".format(age_test))
    print("Predicted QRS Axis {}".format(qrs_predicted_eval))

    # print metrics
    print('MAE for the Adiabatic Linear Regression = ', qrs_lin_regression.mae(qrs_test, qrs_predicted_eval))
    print('MSE for the Adiabatic Linear Regression = ', qrs_lin_regression.mse(qrs_test, qrs_predicted_eval))
    print('Root MSE for the Adiabatic Linear Regression = ', qrs_lin_regression.root_mse(qrs_test, qrs_predicted_eval))
    print('R^2 on test dataset for the Adiabatic Linear Regression = ', qrs_lin_regression.r_score(qrs_test, qrs_predicted_eval))
    print('R^2 on test dataset for the Sklearn Linear Regression = ', qrs_lin_regression.r_score(qrs_test, qrs_classical))

    plt.scatter(age_test, qrs_test)
    plt.plot(age_test, qrs_classical, color='pink', linewidth=2, label='Classical')
    plt.plot(age_test, qrs_predicted_eval, color='green', linewidth=3, label='Quantum')
    plt.legend(loc='upper left')
    plt.title("QRS Axis (in degrees) Depending on Age")
    plt.savefig("results/sklearn_vs_adiabatic_test.png")
    plt.show()