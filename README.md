# RHITA (Real-time Hazard Identification and Tracking Algorithm)

## Objective
RHITA is a Python-based algorithm designed for the detection and tracking of hazards in space and time. It processes historical databases of hazard-related variables, such as temperature or precipitation, to identify and analyze extreme events like heatwaves, droughts, or heavy precipitation. The tool provides outputs in various formats, including catalogues, event tracking files, and 3D event representations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gcazzaniga/rhita.git
   cd RHITA
   ```

2. Ensure you have Python 3.13 or higher installed. You can check your Python version with:
   ```bash
   python --version
   ```

3. Install the required dependencies:
   ```bash
   pip install .
   ```

4. (Optional) For development purposes, install additional dependencies for testing:
   ```bash
   pip install pytest
   ```

## How to Launch

1. Prepare a configuration file (e.g., `test_config.ini`) with the desired settings for hazard detection and tracking. An example configuration file is provided in `tests/input/test_config.ini`.

2. Run the RHITA algorithm using the following command:
   ```bash
   rhita --configfile <path-to-config-file>
   ```
   Replace `<path-to-config-file>` with the path to your configuration file.

3. Outputs will be saved in the directory specified in the configuration file under the `[directories]` section.

## Repository Structure

### Root Directory
- **pyproject.toml**: Contains project metadata and dependencies.
- **README.md**: This file, providing an overview of the project.
- **uv.lock**: Lock file for dependency management.

### `src/rhita/`
- **`__init__.py`**: Initializes the RHITA package and provides the main entry point for running the algorithm.
- **create_cmap.py**: Contains functions for creating custom color maps for visualizations.
- **data_processing.py**: Handles data import, subsetting, and preprocessing.
- **hazards_detection.py**: Implements the core logic for detecting and tracking hazards in space and time.
- **main.py**: Orchestrates the execution of the RHITA algorithm, integrating all components.
- **output_options.py**: Manages the generation and saving of output files, including catalogues and event tracking data.
- **plot.py**: Provides functions for visualizing hazard events and their spatiotemporal tracks.
- **statistics.py**: Includes statistical analysis functions for trends and distributions of hazard events.
- **utils.py**: Utility functions, including logging and execution time tracking.

### `tests/`
- **test_run.py**: Contains a test script to validate the functionality of the RHITA algorithm.
- **input/**: Directory for input files, including example configuration files and climate variable datasets.
- **output/**: Directory for storing test outputs, including catalogues and event tracking files. This folder is created after running the tests.

### Example Input
- **`tests/input/climate_variables/ERA5_tas_day_Europe_20220801-20220831.nc`**: Example input dataset for testing.