"""Disable unused pydicom examples in the frozen browser, before import."""

import sys
import types

examples = types.ModuleType("pydicom.examples")
examples.__doc__ = "Example datasets are not distributed in the CMVD browser."
sys.modules["pydicom.examples"] = examples
