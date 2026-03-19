import numpy as np
import traceback

print("np.dot(np.zeros(5), np.zeros(7)):")
try:
    np.dot(np.zeros(5), np.zeros(7))
except Exception as e:
    print(repr(e))

print("\nnp.dot(np.zeros(7), np.zeros(5)):")
try:
    np.dot(np.zeros(7), np.zeros(5))
except Exception as e:
    print(repr(e))

print("\nTesting DirectionalEngine initialization...")
from core.directional_engine import DirectionalEngine
de = DirectionalEngine(5)
print("de._regime_p_up.shape:", de._regime_p_up.shape)
print("de._regime_p_down.shape:", de._regime_p_down.shape)
