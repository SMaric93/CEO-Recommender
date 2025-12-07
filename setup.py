from setuptools import setup, find_packages

setup(
    name="ceo_firm_matching",
    version="0.1.0",
    description="Two Tower Recommender System for CEO-Firm Matching",
    author="Gemini Agent",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "shap"
    ],
    entry_points={
        'console_scripts': [
            'ceo-firm-match=ceo_firm_matching.cli:main',
        ],
    },
    python_requires=">=3.8",
)
