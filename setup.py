from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='LymphoMNIST',
    version='1.0',
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
    description='A comprehensive dataset for lymphocyte image classification (B cells, T4 cells, and T8 cells).',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='pytorch dataset lymphocyte classification CNN MNIST biomedical imaging machine-learning',
    url='https://github.com/Khayrulbuet13/LymphoMNIST',
    project_urls={
        'Bug Reports': 'https://github.com/Khayrulbuet13/LymphoMNIST/issues',
        'Source': 'https://github.com/Khayrulbuet13/LymphoMNIST',
        'Documentation': 'https://github.com/Khayrulbuet13/LymphoMNIST/blob/main/README.md',
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    python_requires='>=3.6',
    license='Apache License 2.0',
)
