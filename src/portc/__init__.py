"""Copyright (c) 2025 Algorithmiq Development Team. All rights reserved.

portc: Parallel Out-of-Core Randomized Tensor-Network Compression.
"""

from __future__ import annotations

import logging

from ._plan import Resources
from ._tensor_train import SiteLike
from ._version import version as __version__
from ._version import version_tuple as __version_tuple__
from .apply import apply
from .compress import compress
from .stack import src

# Stay silent unless the application configures logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Resources",
    "SiteLike",
    "__version__",
    "__version_tuple__",
    "apply",
    "compress",
    "src",
]
