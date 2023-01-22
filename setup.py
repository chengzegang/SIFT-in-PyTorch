from setuptools import find_packages, setup

setup(
    name="torchsift",
    version="0.1.0",
    description="A Pytorch implementation of sift brute-force matcher and RANSAC",
    author="Zegang Cheng",
    author_email="zc2309@nyu.edu",
    url="https://github.com/chengzegang/TorchSIFT",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.1",
        "torchvision>=0.13.1",
        "numpy",
        "Pillow",
        "opencv-python",
        "setuptools",
    ],
)
