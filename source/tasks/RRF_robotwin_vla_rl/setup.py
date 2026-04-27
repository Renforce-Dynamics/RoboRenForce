from setuptools import find_packages, setup

setup(
    name="RRF_robotwin_vla_rl_tasks",
    version="0.0.0",
    packages=find_packages(),
    author="Zaterval | ZiangZheng",
    maintainer="Ziang Zheng",
    maintainer_email="ziang_zheng@foxmail.com",
    license="BSD-3",
    description=(
        "RoboRenForce VLA-RL task wrappers for RoboTwin. "
        "Registers gym task IDs that bundle env_cfg + algo runner cfg, "
        "mirroring the locomotion (RRF_mjlab) pattern."
    ),
    python_requires=">=3.10",
    install_requires=[
        # Core deps; benchmark sim deps (sapien, mplib, RoboTwin) are install-on-need.
        "torch>=2.7.0",
        "numpy>=1.16.4",
        "gymnasium>=0.29",
    ],
)
