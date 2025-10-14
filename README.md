# LymphoMNIST

Introducing LymphoMNIST, a comprehensive dataset tailored for the nuanced classification of lymphocyte images, encompassing an extensive collection of approximately 80,000 high-resolution 64x64 images. This dataset meticulously categorizes lymphocytes into three primary classes: B cells, T4 cells, and T8 cells, thereby facilitating a focused study on these fundamental immune cell types. Designed with precision, LymphoMNIST stands out as a high-quality, MNIST-like repository of standardized biomedical imagery, purposefully curated to ensure immediate applicability without necessitating extensive background knowledge in biomedical imaging.

LymphoMNIST aims to bridge the gap in biomedical image analysis by providing a dataset that is not only vast in scale but also rich in detail, thereby supporting a wide array of research endeavors, from fundamental biological studies to advanced computational model development. This dataset is particularly curated to cater to a variety of classification challenges, ranging from straightforward binary classifications to more complex multi-class identifications, thereby offering an invaluable resource for both educational and research-based explorations in the fields of biomedical image analysis, computer vision, and machine learning.

By focusing exclusively on B, T4, and T8 lymphocyte cells at a high resolution, LymphoMNIST enables the detailed study and development of specialized models that can accurately distinguish between these cell types, a critical requirement in both clinical diagnostics and immunological research. Whether for academic purposes, algorithm benchmarking, or enhancing the capabilities of open-source and commercial AutoML tools, LymphoMNIST provides a foundational platform for innovation and advancement in the medical imaging domain.

## File structure 
LymphoMNIST is organized as follows for easy integration into your projects:


```txt
LymphoMNIST/
├── LymphoMNIST/
│   ├── __init__.py
│   ├── LymphoMNIST.py  # main class for loading the data
│   └── utils.py        # Additional utility functions for easier visualization
├── tests/
│   ├── __init__.py
│   └── test_LymphoMNIST.py  # Optional: tests for the package
├── setup.py
├── README.md
└── LICENSE
```

<!-- Getting Started with LymphoMNIST -->
## Getting Started with LymphoMNIST

Get up and running with the LymphoMNIST dataset in a few simple steps. This guide will walk you through installing necessary dependencies, setting up the LymphoMNIST package, and loading the dataset for use in your machine learning models.

### Step 1: Install Dependencies
```bash
pip install torch torchvision Pillow numpy tqdm requests matplotlib
```

### Step 2: Install LymphoMNIST package
```bash
pip install LymphoMNIST
```

### Step 3: Check LymphoMNIST version
```python
import LymphoMNIST as info
print(f"LymphoMNIST v{info.__version__} @ {info.HOMEPAGE}")
```

For a detailed tutorial on using LymphoMNIST,  follow this Google Colab notebook. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Khayrulbuet13/LymphoMNIST/blob/main/examples/tutorial.ipynb)



### Built With

* [![PyTorch][PyTorch-shield]][PyTorch-url]
* [![NumPy][numpy-shield]][numpy-url]
* [![Matplotlib][matplotlib-shield]][matplotlib-url]
* [![tqdm][tqdm-shield]][tqdm-url]


<!-- LICENSE -->
## License

Distributed under the Apache License. See `LICENSE` for more information.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)](./LICENSE)





<!-- CONTACT -->
## Contact

Khayrul Islam - [@LinkedIN](https://linkedin.com/in/khayrulbuet13) - khayrulbuet13@alum.lehigh.edu

Project Link: [LymphoMNIST](https://github.com/Khayrulbuet13/LymphoMNIST)



<!-- ACKNOWLEDGMENTS -->
<br><br>
## Acknowledgments

This project is funded by:

![NSF](Images/NSF.jpeg)




<!-- Release Note -->

## Release Note

*Initial release: This is the first release of LymphoMNIST, marking the introduction of the dataset to the research community.*

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- MARKDOWN LINKS & IMAGES -->


<!-- Linkedin -->
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-blue?logo=linkedin

[linkedin-url]: https://linkedin.com/in/khayrulbuet13


<!-- Pytorch -->
[PyTorch-shield]:https://img.shields.io/static/v1?style=for-the-badge&message=PyTorch&color=EE4C2C&logo=PyTorch&logoColor=FFFFFF&label=

[PyTorch-url]:https://pytorch.org


<!-- NumPy -->
[NumPy-shield]: https://img.shields.io/static/v1?style=for-the-badge&message=NumPy&color=013243&logo=NumPy&logoColor=FFFFFF&label=

[NumPy-url]: https://numpy.org

<!-- tqdm -->
[tqdm-shield]:  https://img.shields.io/static/v1?style=for-the-badge&message=tqdm&color=222222&logo=tqdm&logoColor=FFC107&label=

[tqdm-url]: https://tqdm.github.io


<!-- Matplotlib -->
[Matplotlib-shield]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org

<!-- Linkedin -->


<!-- Linkedin -->