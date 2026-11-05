"""Packaging metadata for OneID."""

from setuptools import find_packages, setup

setup(
    name="OneID",
    version="0.1.0",
    description=(
        "Agent-based interpretation of paired optical/SAR remote-sensing imagery "
        "with shared-identity cross-modal reasoning"
    ),
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
)
