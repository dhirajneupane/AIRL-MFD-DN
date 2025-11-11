# AIRL-MFDD-DN
## About
This is reseach work based on Machinary Fault detection.
## Requirements

- Python >= 3.8
- pip >= 20.0
- NumPy 2.0.2
- PyTorch 2.5.1,
- Matplotlib 3.9.4
- SciPy 1.13.
```
pip install -r requirements.txt
```
## Installation

1. Clone this repository:
   ```bash
   cd AIRL-MFDD-DN-May2025
   ```

**The dataset is provided in the "data" folder. The ./data/AIRL folder contains data for the AIRL framwork which can be run in the file 'AIRL_sensorRF2.ipynb'. Data for the contextual bandits (contextualBandit_RF2.ipynb) can be found in ./data/Contextual_bandit.**
The dataset provided in this repo is already preprocessed. _Only the HUMS2023 processed dataset is provided here because of the space issue. _

If you want to work with the original dataset provided by HUMS2023 Committee, please refer to [this link](https://www.dst.defence.gov.au/our-technologies/helicopter-main-rotor-gearbox-planet-gear-fatigue-crack-propagation-test). The code to preprocess the original data is provided in 'HUMS2023_preprocessing.m' file. **Remember** this file is helpful if you are working with the original dataset provided by HUMS2023 committee.


For IMS dataset, one can visit [this site](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset?select=1st_test) and for XJTU_SY dataset, please refer to [this website](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets).
The codes for AIRL and baselines used in this research are available here.
