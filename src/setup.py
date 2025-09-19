from setuptools import setup, find_packages

setup(
    name="automotive_defect_prediction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "imbalanced-learn",
        "scipy",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "joblib",
    ],
)
