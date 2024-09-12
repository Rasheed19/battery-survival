from setuptools import find_packages, setup

setup(
    name="survival_project",
    version="0.0.1",
    author="Rasheed Ibraheem",
    author_email="R.O.Ibraheem@sms.ed.ac.uk",
    maintainer="Rasheed Ibraheem",
    maintainer_email="R.O.Ibraheem@sms.ed.ac.uk",
    description="Survival analysis of lithium-ion batteries",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Rasheed19/battery-survival.git",
    project_urls={
        "Bug Tracker": "https://github.com/Rasheed19/survival-project.git/issues"
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires="<=3.10.15",
)
