from setuptools import setup

setup(
    name="muxsim",
    version="0.1",
    description="A tool to simulate cell/guide matrices",
    author="Noam Teyssier",
    author_email="Noam.Teyssier@ucsf.edu",
    packages=["muxsim"],
    install_requires=["numpy", "scipy", "pandas", "seaborn", "matplotlib", "tqdm"],
)
