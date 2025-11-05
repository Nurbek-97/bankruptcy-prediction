from setuptools import setup, find_packages

setup(
    name="bankruptcy-prediction",
    version="1.0.0",
    author="Nurbek Xalimjonov",
    author_email="nurbekkhalimjonov070797@gmail.com",
    description="Advanced bankruptcy prediction using ML and econometrics",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bankruptcy-prediction",
    packages=find_packages(),
    classifiers=[   
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
        "catboost>=1.2.2",
        "statsmodels>=0.14.0",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
        "tqdm>=4.66.1",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "black>=23.9.1",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
    },
)