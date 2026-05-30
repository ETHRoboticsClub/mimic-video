"""Pytest conftest — inject mock decord so process_lerobot can import on platforms without it."""

import sys
from data_preprocessing.action.tests import mock_decord

sys.modules["decord"] = mock_decord
