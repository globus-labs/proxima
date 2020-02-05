"""Utility operations for the Monte carlo application"""

from gitinfo import get_git_info
import platform
import requests
import os


_qm9_url = "https://github.com/globus-labs/g4mp2-atomization-energy/raw/master/data/output/g4mp2_data.json.gz"
_qm9_path = os.path.join(os.path.dirname(__file__), 'data', 'qm9.json.gz')


def get_qm9_path() -> str:
    """Get the path to the QM9 dataset"""
    if not os.path.isfile(_qm9_path):
        _download_data()
    return _qm9_path


def _download_data():
    """Download the QM9 data"""

    # Make sure the data path is available for saving
    data_dir = os.path.dirname(_qm9_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download and save the file
    req = requests.get(_qm9_url, stream=True)
    with open(_qm9_path, 'wb') as fp:
        for chunk in req.iter_content(1024 ** 2):
            fp.write(chunk)


def get_platform_info():
    """Get information about the computer running this process"""

    if hasattr(os, 'sched_getaffinity'):
        accessible = len(os.sched_getaffinity(0))
    else:
        accessible = os.cpu_count()
    return {
        'git_commit': get_git_info()['commit'],
        'processor': platform.machine(),
        'python_version': platform.python_version(),
        'python_compiler': platform.python_compiler(),
        'hostname': platform.node(),
        'os': platform.platform(),
        'cpu_name': platform.processor(),
        'n_cores': os.cpu_count(),
        'accessible_cores': accessible,
    }
