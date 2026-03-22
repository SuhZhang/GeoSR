import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import torch
except ImportError:
    pass

__version__ = '0.2rc1'
