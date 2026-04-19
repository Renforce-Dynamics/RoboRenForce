from setuptools import find_packages, setup

setup(
    name="RRF_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "RoboRenForce",
        "torch",
    ],
    extras_require={
        "qwen2vl": ["transformers>=4.40", "peft>=0.10", "einops"],
        "openpi": ["transformers>=4.40", "peft>=0.10"],
        "all": ["transformers>=4.40", "peft>=0.10", "einops"],
    },
)
