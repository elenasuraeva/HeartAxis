# Authors
# Olga Okrut. GitHub: olgOk
# Elena Suraeva. GitHub: elenasuraeva

import pandas as pd
import numpy as np
import json
from typing import Tuple, Any

class QuantumHeart(): 
    """Implementation of heart on Quantum Computer."""

    def __init__(self, device):
        self.device = device

    def get_qrs_axis(self, ecg_i:np.array, ecg_iii:np.array) -> int:
        """Calculates QRS Axis using IBM Qiskit.
            Args:
                ecg_i: (numpy array) Lead I of a heart.
                ecg_i: (numpy array) Lead III of a heart.
            Returns:
                heart_axis: (int), the value of heart axis in degrees.
                None if ECG data is incorrect.
        """

        # error hanglings
        if not np.size(ecg_i) and not np.size(ecg_iii):
            raise ValueError('Arrays for heart leads are empty.')
        if np.size(ecg_i) != np.size(ecg_iii):
            raise ValueError("Lead I and Lead III arrays have different dimension.")

        # prepare data to map to a qubit
        peaks, _ = find_peaks(ecg_i, prominence=500)
        peak_idx = 8

        # if issue in dataset occures
        try:
            I = ecg_i[peaks[peak_idx]-50:peaks[peak_idx]+50]
            III = ecg_iii[peaks[peak_idx]-50:peaks[peak_idx]+50]
        except:
            return None
        
        delta_I = np.mean(np.concatenate((I[0:20], I[80:100])))
        delta_III = np.mean(np.concatenate((III[0:20], III[80:100])))
        I = I - delta_I
        III = III - delta_III
        V_I = abs(np.max(I)) - abs(np.min(I))
        V_III = abs(np.max(III)) - abs(np.min(III))
        
        # calculate heart axis on a quantum device (gate-based)
        value = []
        for i in range(360):
            qc_I = QuantumCircuit(1)
            qc_III = QuantumCircuit(1)

            qc_I.h(0)
            qc_I.p(i*np.pi/180, 0)
            qc_I.ry(-0.0001*V_I, 0)

            qc_III.h(0)
            qc_III.p(i*np.pi/180, 0)
            qc_III.p(-np.pi/6, 0)
            qc_III.rx(-0.0001*V_III, 0)

            job_I = execute(qc_I, self.device)
            result_I = job_I.result()
            statevector_I = result_I.get_statevector()

            job_III = execute(qc_III, self.device)
            result_III = job_III.result()
            statevector_III = result_III.get_statevector()
            value.append(abs(statevector_I[0])**2+abs(statevector_III[1])**2)

        qrs_value = np.argmax(value)
        return int(qrs_value)

def gather_ecg_data():
    """
        Gather and analyze around 40,000 ECG files.
        Stores QTc, QRS Axis, QRS Duration and Age in .json file.   
    """

    directory = "dataset/ECGDataDenoised/"
    df = pd.read_csv('dataset/WavelenghtHere.csv')
    df_end = pd.DataFrame(columns=['Age', 'QTc', 'QRSDuration', 'QRSAxis'])
    quantum_divice = Aer.get_backend('statevector_simulator')

    data_to_dump = {}

    counter = 0 
    for index in df.index:

        # print("Iteration {} out of {}". format(counter, 10647))

        file_name = df['FileName'][index]
        colnames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        df_to_process = pd.read_csv(directory + file_name + '.csv', names=colnames)

        # extract lead I and lead III from data file
        ecg_i = np.asarray(df_to_process["I"])
        ecg_iii = np.asarray(df_to_process["III"])

        # calculate QRS Axis using IBM Quantum device
        q_heart = QuantumHeart(quantum_divice)
        qrs_axis = q_heart.get_qrs_axis(ecg_i, ecg_iii)

        if qrs_axis == None:
            counter += 1
            continue

        data_to_dump['Age'] = int(df['PatientAge'][index])
        data_to_dump['QTc'] = int(df['QTCorrected'][index])
        data_to_dump['QRSDuration'] = int(df['QRSDuration'][index])
        data_to_dump['QRSAxis'] = qrs_axis

        with open('dataset/data_for_training.json', mode='a') as f:
            json.dump(data_to_dump, f, ensure_ascii=True, indent="", separators=(",", ":"))

        counter += 1

def get_qrs_axis(df_end):
    """
    Calcualtes age and QRS Axis for different age groups. 
    The result is the mean value for each age group.

    Args: 
        df_end: a pandas dataframe with the records.
    
    Returns:
        age: a sorted np.array with the patient ages
        qrsaxis: a sorted np.array with the patient measurements on QRS axis.
    """
    df_normal = pd.DataFrame(columns=[ 'Age', 'QTc', 'QRSDuration', 'QRSAxis'])
    for index in df_end.index:
        if df_end["QRSAxis"][index] in range(-30, 90):
            df_normal = df_normal.append( df_end.iloc[[index]], ignore_index=True)
    
    df_normal.sort_values(by=["Age"], ascending=True, inplace=True)
    df_normal.drop(df_normal[df_normal["Age"] <=14 ].index, inplace=True)

    age_bygroup = []
    qrsaxis_byage = []

    for age_group in range(15, 91, 5):
        age_bygroup.append(age_group+2)
        mean_qrs = df_normal.loc[(df_normal["Age"] >=age_group) & (df_normal["Age"] <= age_group+4), "QRSAxis"].mean()
        qrsaxis_byage.append(mean_qrs)

    return np.asarray(age_bygroup), np.asarray(qrsaxis_byage)

def get_qrs_duration(df_end):
    """
    Calcualtes age and QRS Duration for different age groups. 
    The result is the mean value for each age group.

    Args: 
        df_end: a pandas dataframe with the records.
    
    Returns:
        age: a sorted np.array with the patient ages
        qrs_duration: a sorted np.array with the patient measurements on QRS duration.
    """
    df_normal = pd.DataFrame(columns=[ 'Age', 'QTc', 'QRSDuration', 'QRSAxis'])

    for index in df_end.index:
        if df_end["QRSDuration"][index] in range(80, 110):
            df_normal = df_normal.append( df_end.iloc[[index]], ignore_index=True)

    df_normal.sort_values(by=["Age"], ascending=True, inplace=True)
    df_normal.drop(df_normal[df_normal["Age"] <=14 ].index, inplace=True)

    age_bygroup = []
    qrsduration_byage = []

    for age_group in range(15, 91, 5):
        age_bygroup.append(age_group+2)
        mean_qrs_dur = df_normal.loc[(df_normal["Age"] >=age_group) & (df_normal["Age"] <= age_group+4), "QRSDuration"].mean()
        qrsduration_byage.append(mean_qrs_dur)

    return np.asarray(age_bygroup), np.asarray(qrsduration_byage)
  

def get_qts(df_end):
    """
    Calcualtes age and Qtc for different age groups. 
    The result is the mean value for each age group.
    Args: 
        df_end: a pandas dataframe with the records.
    
    Returns:
        age: a sorted np.array with the patient ages
        qtc: a sorted np.array with the patient measurements on QTC axis.
    """
    df_normal = pd.DataFrame(columns=[ 'Age', 'QTc', 'QRSDuration', 'QRSAxis'])
    for index in df_end.index:
        if df_end["QTc"][index] in range(360, 441):
            df_normal = df_normal.append( df_end.iloc[[index]], ignore_index=True)
    
    df_normal.sort_values(by=["Age"], ascending=True, inplace=True)
    df_normal.drop(df_normal[df_normal["Age"] <=14 ].index, inplace=True)

    age_bygroup = []
    qtc_byage = []
    for age_group in range(15, 91, 5):
        age_bygroup.append(age_group+2)
        mean_qtc = df_normal.loc[(df_normal["Age"] >=age_group) & (df_normal["Age"] <= age_group+4), "QTc"].mean()
        qtc_byage.append(mean_qtc)
    
    return np.asarray(age_bygroup), np.asarray(qtc_byage)


def get_data_for_training(parameter:str):
    """
        Prepares the data for classical ML training or for training as Addiabatic Linear Regression.
        The data is grouped and analized by age groups as described in the article
        https://pubmed.ncbi.nlm.nih.gov/35807018/ .
        Normal QRS_axis is in range of -30 to +90 degrees.
        Normal QRS Duration is in between 0.08 and 0.10 seconds (or from 80 to 100 milliseconds).
        Normal QTc duration is in range of 0.36 to 0.44 seconds (or from 360 to 440 miliseconds).
        Args:
            parameter: a string of a heart charachteristic - QRSAxis, QRSDuration, QTc.
        Returns:
            age: a np.array with patient ages
            heart parameter: a np.array containing any of QRSAxis, or QRSDuration, or QTc.
    """

    with open('dataset/train_data.json') as file:
        data = json.load(file)

    df_end = pd.DataFrame.from_dict(data["data"])
    
    # convert ECG axis from -30 deg to 180 deg.
    for index in df_end.index:
        if df_end["QRSAxis"][index] in range(0, 180):
            df_end["QRSAxis"][index] = -df_end["QRSAxis"][index]
        elif df_end["QRSAxis"][index] in range(180, 360):
            df_end["QRSAxis"][index] = abs(360 - df_end["QRSAxis"][index])

    if parameter == "QRSAxis":
        age, qrs_axis = get_qrs_axis(df_end)
        return age, qrs_axis
    elif parameter == "QRSDuration":
        age, qrs_duration = get_qrs_duration(df_end)
        return age, qrs_duration
    elif parameter == "QTc":
        age, qtc = get_qts(df_end)
        return age, qtc
    else:
        raise ValueError("Invalid parameter. Select from QRSAxis, QRSDuration, QTc.")

if __name__ == "__main__":

    # gather_ecg_data()

    # get_data_for_training()
    pass
