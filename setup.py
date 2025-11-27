from setuptools import setup

setup(
    name="ceo_firm_matching",
    version="0.1.0",
    description="Two Tower Recommender System for CEO-Firm Matching",
    author="Gemini Agent",
    py_modules=["two_towers"],
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "shap"
    ],
    python_requires=">=3.8",
)
