import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LandmarkDetector-chichichkin", # Replace with your own username
    version="0.0.1",
    author="Alexander Chagochkin",
    author_email="chichichkin@yandex.ru",
    description="Small SAN Landmark Detector wrap",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chichichkin/LandmarkDetector",
    project_urls={
        "Bug Tracker": "https://github.com/Chichichkin/LandmarkDetector/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "landmark_detector"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)