"""Setup script for the FalconMOT package.

Most metadata lives in ``pyproject.toml``; this file is kept for editable
installs (``pip install -e .``) on older tooling.
"""
from setuptools import find_packages, setup


def _read_requirements():
    reqs = []
    with open("requirements.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                reqs.append(line)
    return reqs


def _read_readme():
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


setup(
    name="falconmot",
    version="1.0.0",
    description="DINOv3-backbone JDE tracker for multi-object tracking in drone video",
    long_description=_read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["falconmot", "falconmot.*"]),
    python_requires=">=3.9",
    install_requires=_read_requirements(),
    extras_require={
        "onnx": ["onnx>=1.15", "onnxruntime>=1.16"],
        "sparse": ["xformers>=0.0.23"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
