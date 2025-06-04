# Battery Storage Optimization

This repository contains Python scripts to optimize the operation of a battery energy storage system (BESS) for day-ahead and intraday electricity markets. The main module `V9_optimizer.py` builds a Pyomo model that schedules charge and discharge power to maximise trading revenue while respecting efficiency losses, cycle limits and state-of-charge constraints.

`V9_run_optimizer` demonstrates how to use the optimizer with price data read from Excel workbooks. Results can be exported back to Excel along with charts for further analysis.

## Installation
1. Clone the repository.
2. Install the required packages with
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare Excel files containing day-ahead and intraday price series.
2. Edit the paths to these files in `V9_run_optimizer`.
3. Run the example script:
   ```bash
   python V9_run_optimizer
   ```
   Output spreadsheets will be written to the `ergebnisse` folder.

## Repository structure
- `V9_optimizer.py` – optimization class based on Pyomo.
- `V9_run_optimizer` – example script that loads price data, runs the model and exports results.
- `requirements.txt` – list of Python dependencies.

The scripts are provided as a starting point and may need adjustments (e.g. missing `visualizer` helpers) to match your data and analysis workflow.

