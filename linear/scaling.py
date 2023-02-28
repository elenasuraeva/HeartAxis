# Maintainer: Olga Okrut. GitHub: olgOk
# This code is the part of the Adiabatic-QML by DarkStarQuantumLab
# The source code: https://github.com/DarkStarQuantumLab/Adiabatic-QML

from typing import Tuple, Any
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, StandardScaler

class Scaling():
    """Implementation of data entries re-scaling on sklearn.preprocessing module.
    """

    def __init__(self, scaler:str = "StandardScaler"):
        self.scaler_type = scaler

    def scale(self, training_data:dict):
        """Scales the data entries based on the scaler type provided.
            Supported scalers: StandardScaler, MaxAbsScaler, RobustScaler.

            Args:
                training_data: a dictionary of train, label points.
            Return:
                rescaled: a dictionary of the re-scaled train, label points.
                scaling_factor: int, a scaling factor. 
        """
        if self.scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif self.scaler_type == "MaxAbsScaler":
            scaler = MaxAbsScaler()
        elif self.scaler_type == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise Warning("f{self.scaler_type} is not supported. \
                The default sclaer StandardScaler will be used.")
            scaler = StandardScaler()
        rescaled = {}
        for key, value in training_data.items():
            rescaled[key] = scaler.fit_transform(value)
        
        scaling_factor = scaler.scale_
        return rescaled, scaling_factor