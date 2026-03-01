"""
Root conftest.py for ai-drone-auto-vehicle test suite.

Adds the src/ directory to sys.path so all modules are importable
without installing the package.
"""
import sys
import os

# Ensure src/ is on the path for all test files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
