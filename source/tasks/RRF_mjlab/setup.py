from setuptools import find_packages, setup

setup(
    name="RRF_mjlab_tasks",
    version="0.0.0",
    packages=find_packages(),
    author="Zaterval | ZiangZheng",
    maintainer="Ziang Zheng",
    maintainer_email="ziang_zheng@foxmail.com",
    url="https://github.com/mujocolab/mjlab",
    license="BSD-3",
    description="RoboRenForce task wrappers for MJLab (MuJoCo Warp) locomotion environments",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7.0",
        "numpy>=1.16.4",
        "mujoco>=3.7.0",
        "mujoco-warp>=3.7.0.1",
        "warp-lang>=1.12.0",
    ],
)
