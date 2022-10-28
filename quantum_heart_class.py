# Elena Suraeva. Email: elenasuraeva21@gmail.com

from qiskit import QuantumRegister, QuantumCircuit, execute
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks
from qiskit import QuantumCircuit, execute, Aer
from random import randint


class QuantumHeart: 
    """Implementation of heart on Quantum Computer.
       Default device simulator is a statevector simulator. 
    """

    def __init__(self, device=Aer.get_backend('statevector_simulator')):
        self.device = device
        
    def _cost(self, phi:float, V_I:np.array, V_aVF:np.array) -> float:
        """ Calculate the probabilty of measurements in a defined basis.
            The maximum value of means phi is a desired angle.
            Args:
                phi: (float) an angle.
                V_I: (numpy array) Lead I of a heart.
                V_aVF: (numpy array) Lead aVF of a heart
            Returns:
                probabiluty: (float) Probability in a defined basis. 
        """
        qc = QuantumCircuit(2)
        qc.h([0,1])
        qc.p(phi, [0,1])
        qc.ry(-1e-04*np.sqrt(3)*V_I, 0)
        qc.rx(-1e-04*2*V_aVF, 1)
        job = execute(qc, self.device)
        result = job.result()
        statevector = result.get_statevector()

        probability = abs(statevector.data)
        return probability

    def get_qrs_axis(self, ecg_i:np.array, ecg_avf:np.array) -> int:
        """Calculates QRS Axis using IBM Qiskit.
            Args:
                ecg_i: (numpy array) Lead I of a heart.
                ecg_avf: (numpy array) Lead aVF of a heart.
            Returns:
                heart_axis: (int), the value of heart axis in degrees.
                None if ECG data is incorrect or corrupt.
        """

        # error hanglings
        if not np.size(ecg_i) and not np.size(ecg_avf):
            raise ValueError('Arrays for heart leads are empty.')
        if np.size(ecg_i) != np.size(ecg_avf):
            raise ValueError("Lead I and Lead aVF arrays have different dimension.")

        # prepare data to map to a qubit
        peaks, _ = find_peaks(ecg_i, prominence=500)
        if peaks.size == 0:
            peaks, _ = find_peaks(-ecg_i, prominence=500)
        if peaks.size == 0:
            peaks, _ = find_peaks(ecg_avf, prominence=500)
        if peaks.size == 0:
            peaks, _ = find_peaks(-ecg_avf, prominence=500)
        peak_idx = randint(0, len(peaks))

        # if issue in dataset occures, return None
        try:
            I = ecg_i[peaks[peak_idx]-50:peaks[peak_idx]+50]
            aVF = ecg_avf[peaks[peak_idx]-50:peaks[peak_idx]+50]
        except:
            return None
        
        ref_I = np.mean(np.concatenate((I[0:20], I[80:100])))
        ref_aVF = np.mean(np.concatenate((aVF[0:20], aVF[80:100])))
        I = I - ref_I
        aVF = aVF - ref_aVF
        V_I = abs(np.max(I)) - abs(np.min(I))
        V_aVF = abs(np.max(aVF)) - abs(np.min(aVF))
        
        # calculate heart axis on a quantum device (gate-based)
        angle = np.pi/4
        c = self._cost( angle, V_I, V_aVF)
        k0 = (c[0]+c[2])/(c[1]+c[3])
        k1 = (c[0]+c[1])/(c[2]+c[3])
        if k0 > 1 and k1 <= 1:
            angle = angle
        elif k0 <= 1 and k1 < 1:
            angle += np.pi/2
        elif k0 < 1 and k1 >= 1:
            angle -= np.pi
        elif k0 >= 1 and k1 > 1:
            angle -= np.pi/2
        step = np.pi/180
        if (self._cost( angle + step/2, V_I, V_aVF) - self._cost( angle - step/2,  V_I, V_aVF))[2] < 0:
            step *= -1
        while self._cost( angle + step, V_I, V_aVF)[2] > self._cost(angle, V_I, V_aVF)[2]:
            angle += step
        qrs_value = angle * 180/np.pi

        return int(qrs_value)