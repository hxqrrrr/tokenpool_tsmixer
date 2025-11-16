"""
Models module for RUL prediction
Only PhasePoolTokenMixer model is maintained in this framework
"""
from .base_model import BaseRULModel
from .phasepool_tokenmixer import PhasePoolTokenMixerRUL

__all__ = ['BaseRULModel', 'PhasePoolTokenMixerRUL']
