from setuptools import setup, find_packages

setup(
    name="muse_toolbox",
    version="0.1.0",
    description="Multi-Source Enhancement Toolbox",
    author="Henri Gode",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "torch",
        "torchaudio",
        "hydra-core",
        "python-dotenv",
        "wandb",
        "lightning",
    ],
)