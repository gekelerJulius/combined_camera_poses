# Bachelor-Thesis Julius Gekeler
## Installation
1. Install [Python 3.8] (https://www.python.org/downloads/release/python-380/)
2. Create a virtual environment with `python3 -m venv venv` or conda
3. Activate the virtual environment with `source venv/bin/activate` or conda
4. Install the requirements with `pip install -r requirements.txt`
## Usage
1. Choose from Experiment 1, 2 or 3
2. Copy the contents of your experiments folder into the `simulation_data` folder
3. Run `python app.py`

If any problems occur, please contact me at gekelerjulius@gmail.com

## Structure
- `app.py` is the main file to run the application
- `simulation_data` contains the data from the experiments
- `classes/person_recorder.py` contains the PersonRecorder class which is used for tracking the persons using a kalman filter
- `classes/record_matcher.py` contains the RecordMatcher class which is used for matching the persons from the different cameras
- `classes/true_person_loader.py` contains the TruePersonLoader class which is used for loading the true persons from the experiments for validation of results
- `functions/funcs.py` contains several helper functions used in the application
- `functions/icp.py` contains the code used for the ICP algorithm and finding the transformation between two point clouds. The ICP Algorithm itself is not used in the final version.
- `functions/estimate_extrinsic.py` contains the code used for estimating the extrinsic parameters of the cameras
- `functions/calc_repr_errors.py` contains the code used for calculating the reprojection errors of the camera calibration
- `functions/compare_persons.py` contains the code used for calculating a difference between two persons
- `functions/calc_repr_errors.py` contains the code used for calculating the reprojection errors of the camera calibration
