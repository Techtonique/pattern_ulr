from setuptools import setup, find_packages

setup(
    name="pattern_ulr",
    version="0.1.0",
    description="Mack and ODP-bootstrap ULR distributions for an immature book using a legacy chain-ladder pattern",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9",
    install_requires=["numpy", "pandas", "scipy", "tqdm", "matplotlib"],
    extras_require={"test": ["pytest", "chainladder"],
                    "examples": ["matplotlib"]},
)