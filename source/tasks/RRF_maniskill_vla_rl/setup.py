from setuptools import find_packages, setup

setup(
    name="RRF_maniskill_vla_rl_tasks",
    version="0.0.0",
    packages=find_packages(),
    description="RoboRenForce VLA-RL task wrappers for ManiSkill.",
    python_requires=">=3.10",
    install_requires=["torch>=2.7.0", "numpy>=1.16.4", "gymnasium>=0.29"],
)
