"""STAMP integration utilities.

This package prepares STAMP configuration files and optionally invokes the
external STAMP command-line application. It does not reimplement STAMP.
"""

from bias_analysis_framework.stamp.preprocess import run_stamp_feature_extraction

__all__ = ["run_stamp_feature_extraction"]