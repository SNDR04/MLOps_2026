from setuptools import setup, find_packages

setup(
    name="ml_core",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "h5py",
        "pyyaml",
        "tqdm",
        "torcheval",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "pandas"
    ],
)
