from typing import Literal
from Configs import ConfigHomeLab
from Configs import ConfigCluster


class Config:
    def __init__(self, env: Literal["HomeLab", "Cluster"]):
        self.__config = None
        if env == "HomeLab":
            self.__config = ConfigHomeLab()
        elif env == "Cluster":
            self.__config = ConfigCluster()

    def get_configuration(self):
        return self.__config
