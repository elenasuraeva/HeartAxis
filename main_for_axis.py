# Elena Suraeva. Email: elenasuraeva21@gmail.com
# Authors: Olga Okrut. Email:vokrut42sv@gmail.com

import os
from quantum_heart_class import QuantumHeart
from scipy.io import loadmat
from scipy.signal import find_peaks

def get_data():
    """Iterate over directory and get ECG signals.
    """
    directory = "dataset/axis_calculation/"

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            ECG = loadmat(f)['val']
            


if __name__ == "__main__":

    directory = "dataset/axis_calculation/"

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            print(f)
            ECG = loadmat(f)['val']
            print("Calculating the Heart Axis on IBM Quantum ...")
            qrs_axis = QuantumHeart()
            qrs_angle = qrs_axis.get_qrs_axis(ECG[0], ECG[5])
            print("Angle Calculated: {}.".format(qrs_angle))