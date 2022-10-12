from setuptools import find_packages, setup

setup(
    name="torchsift",
    version="0.0.1.dev1",
    description="A Pytorch implementation of SIFT BFMatcher and RANSAC",
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
