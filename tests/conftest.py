import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: renders and processes synthetic room video (~10 s per scenario)")
