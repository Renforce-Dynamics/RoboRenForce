from setuptools import find_packages, setup

setup(
    name="RRF_calvin_tasks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "RoboRenForce",
        "numpy",
        "torch",
    ],
    extras_require={
        "sim": [
            "calvin_env",
        ],
    },
)
