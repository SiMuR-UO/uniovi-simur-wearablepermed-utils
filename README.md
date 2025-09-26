<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-utils.svg?branch=main)](https://cirrus-ci.com/github/<USER>/uniovi-simur-wearablepermed-utils)
[![ReadTheDocs](https://readthedocs.org/projects/uniovi-simur-wearablepermed-utils/badge/?version=latest)](https://uniovi-simur-wearablepermed-utils.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/uniovi-simur-wearablepermed-utils/main.svg)](https://coveralls.io/r/<USER>/uniovi-simur-wearablepermed-utils)
[![PyPI-Server](https://img.shields.io/pypi/v/uniovi-simur-wearablepermed-utils.svg)](https://pypi.org/project/uniovi-simur-wearablepermed-utils/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/uniovi-simur-wearablepermed-utils.svg)](https://anaconda.org/conda-forge/uniovi-simur-wearablepermed-utils)
[![Monthly Downloads](https://pepy.tech/badge/uniovi-simur-wearablepermed-utils/month)](https://pepy.tech/project/uniovi-simur-wearablepermed-utils)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/uniovi-simur-wearablepermed-utils)
-->

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

## Description

> Uniovi Simur WearablePerMed Utils

## Schaffolding
Execute PyScaffold command to create the project:
```
$ putup --markdown uniovi-simur-wearablepermed-utils -p wearablepermed_utils \
     -d "Uniovi Simur WearablePerMed Utils." \
     -u https://github.com/SiMuR-UO/uniovi-simur-wearablepermed-utils.git 
```

Create a virtual environment inside for your project and active it:
```
$ python -m venv .venv
$ source .venv/bin/activate
```

Install and upgrade default modules:
```
$ pip install -U pip setuptools setuptools_scm wheel tox
```

Install and upgrade project modules:
```
$ pip install -U numpy pandas scipy openpyxl matplotlib
```

Install module locally
```
$ pip install -e .
```

Save project requirements:
```
$ pip freeze > requirements.txt
```

Build project commands
```
$ tox -e clean
$ tox -e build
$ tox -e docs
$ tox -e publish -- --repository pypi
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
