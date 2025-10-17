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

Install and upgrade tox automation project manager:
```
$ pip install --upgrade tox
```

Install and upgrade project modules:
```
$ pip install -U numpy pandas scipy openpyxl matplotlib
```

## Code and Debugging

Install library modules:
```
$ pip install -r requirements.txt
```

Install module locally for debugg
```
$ pip install -e .
```

Save project requirements:
```
$ pip freeze > requirements.txt
```

## Project management

Project commands for: test, clean, build, generate documentation or publish your library in pypi repository
Don't forget update the version library from **setup.cfg** project build file:

```
$ tox
$ tox -e clean
$ tox -e build
$ tox -e docs
$ tox -e publish -- --repository pypi
```

## Testing

- Convert binary file to csv:
     ```
     $ sensor_bin_to_csv \
     --bin-file /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_C.BIN \
     --csv-file /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_C.csv
     ```

- Segment all sensor csv files for each participant and correct deviation if exists:
     For participant csv files without some deviation error
     ```
     $ csv_to_segmented_activity \
     --csv-file /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_M.csv \
     --excel-activity-log /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_RegistroActividades.xlsx \
     --body-segment M \
     --output /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_seg_M.npz
     ```

     For participant csv files with some deviation error
     ```
     $ csv_to_segmented_activity \
     --csv-file /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_M.csv \
     --excel-activity-log /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_RegistroActividades.xlsx \
     --body-segment M \
     --output /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_seg_M.npz \
     --sample-init 16088120 \
     --start-time 17:40:00
     ```

- Window segmented files for convolution model likes
     The argument **include-not-estructure-data** can be included if you want add not estructure data
     ```
     $ segmented_activity_to_stack \
     --npz-file /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_seg_M.npz \
     --crop-columns 1:7 \
     --window-size 250 \
     --window-overlapping-percent 50 \
     --output /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_tot_M.npz
     ```

- Extract features from windowed segmented files for random forest model likes
     ```
     $ stack_to_features \
     --stack-file /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_tot_M.npz \
     --output /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1053/PMP1053_W1_tot_M_features.npz∫
     ```

- Partial aggregation for each participant datasets
     ```
     $ aggregate_windows_features \
     --dataset-folder /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1003 \
     --ml-models ESANN,RandomForest \
     --ml-sensors thigh,hip,wrist \
     --output-folder /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input/PMP1003
     ```

- Total aggregation for all participant datasets to train models
     ```
     $ model_aggregation \
     --dataset-folder /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/input \
     --output-folder /home/miguel/git/uniovi/simur/uniovi-simur-wearablepermed-utils/data/output \
     --case-id case_sample
     ```

Pipeline:
![Example result](https://github.com/SiMuR-UO/uniovi-simur-wearablepermed-utils/blob/main/images/pretraining_pipeline.png)

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
