from setuptools import setup, find_packages

setup(
    name='LymphoMNIST',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'numpy',
        'Pillow', 
        'tqdm',
        'requests',
    ],
    author='Khayrul Islam',
    author_email='khayrulbuet13@alum.lehigh.edu',
    description='A PyTorch dataset for the LymphoMNIST dataset.',
    keywords='pytorch dataset resnet CNN MNIST FashionMNIST',
    url='https://github.com/Khayrulbuet13/Lympho3-MNIST.git',
)
