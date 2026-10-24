from setuptools import find_packages, setup

# The code is imported as `src.<module>` throughout (scripts, notebooks, tests),
# so the package root is the `src` directory itself.
setup(
    name="afro-fetalnet",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
)
