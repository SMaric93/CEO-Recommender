from setuptools import setup, find_packages

setup(
    name="ceo_firm_matching",
    version="0.31",
    description="Two Tower Recommender System for CEO-Firm Matching",
    author="Gemini Agent",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "shap",
        "statsmodels",
    ],
    extras_require={
        "wrds": ["wrds"],
        "dev": ["pytest", "pytest-cov"],
    },
    entry_points={
        'console_scripts': [
            'ceo-firm-match=ceo_firm_matching.cli:main',
            'structural-distill=ceo_firm_matching.structural_cli:main',
        ],
    },
    python_requires=">=3.8",
)
