from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="noodlings",
    version="0.1.0",
    author="Caitlyn Meeks",
    author_email="caitlyn@example.com",  # TODO: Update with real email
    description="Multi-timescale affective agents with theatrical control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caitlynmeeks/noodlings",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "noodlemush=applications.cmush.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "applications.cmush": [
            "web/*.html",
            "web/*.js",
            "web/*.css",
            "plays/*.json",
            "world/*.json",
        ],
    },
)
