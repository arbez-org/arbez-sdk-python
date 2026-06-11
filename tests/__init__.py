"""Make ``tests`` a real package so test modules can ``from tests._corpus import ...`` cleanly under
mypy --strict.

Pytest discovery is unaffected.
"""
