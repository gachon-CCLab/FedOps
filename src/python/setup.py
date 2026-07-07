from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fedops",
    version="1.1.30.12",
    author="Semo Yang",
    author_email="tpah20@gmail.com",
    description="FL Lifecycle Operations Management Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gachon-CCLab/FedOps.git",
    packages=find_packages(),
    install_requires=[
        "flwr>=1.0.0",
        "flwr[simulation]",
        "fastapi>=0.70.1",
        "uvicorn[standard]>=0.16.0",
        "requests>=2.26.0",
        "pydantic>=1.10.7",
        "numpy==1.26.4",
        "wandb>=0.15.0",
        "pandas>=1.5.3",
        "pyyaml>=5.0",
        "boto3>=1.24.0",
        "hydra-core",
        "numba",
        "tqdm",
        "transformers==4.43.1",
        "trl==0.8.1",
        "scikit-learn",
        "bitsandbytes",
        "peft==0.14.0",
        "grad-cam",
        "optuna>=3.6",
    ],
    entry_points={
        "console_scripts": [
            "fedops=fedops.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
