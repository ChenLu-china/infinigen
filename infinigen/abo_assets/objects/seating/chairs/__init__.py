import numpy as np

from .abo_chair import ABOChairFactory


def random_chair_factory():

    chair_factories = [
        ABOChairFactory
    ]

    return np.random.choice(chair_factories)