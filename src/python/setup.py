from setuptools import setup, find_packages

setup(
    name='fedops',
    version='0.1.2',
    author='Semo Yang',
    author_email='tpah20@gmail.com',
    description='FL Lifecycle Operations Management Platform',
    long_description='Long description of your library',
    url='https://github.com/gachon-CCLab/FedOps.git',
    packages=find_packages(),
    install_requires=[
        'flwr>=1.0.0',
        'fastapi>=0.70.1',
        'uvicorn[standard]>=0.16.0',
        'requests>=2.26.0',
        'pydantic>=1.10.7',
        'tensorflow>=2.7.0',
        'keras>=2.7.0',
        'numpy>=1.23.5',
        'wandb>=0.15.0',
        'pandas>=1.5.3',
        'pyyaml>=5.0',
        'boto3>=1.24.0',
        'numba'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)