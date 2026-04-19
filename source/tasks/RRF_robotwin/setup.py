from setuptools import find_packages, setup

setup(
    name="RRF_robotwin_tasks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "RoboRenForce",
        "numpy",
        "torch",
    ],
    extras_require={
        "sim": [
            "sapien==3.0.1",
            "mplib==0.2.1",
            "gymnasium>=0.29.1",
            "transforms3d",
            "trimesh",
            "open3d",
            "av",
        ],
    },
)
