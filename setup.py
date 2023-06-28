from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="muxsim",
    version="0.1.0",
    author="Noam Teyssier",
    author_email="Noam.Teyssier@ucsf.edu",
    description="A tool to simulate cell/guide matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/noamteyssier/muxsim",
    packages=["muxsim"],
    install_requires=["numpy", "scipy", "pandas", "seaborn", "matplotlib", "tqdm"],
)
