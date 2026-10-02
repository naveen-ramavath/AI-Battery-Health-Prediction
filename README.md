# AI-Battery-Health-Prediction
Implementation of AI-based Lithium-ion Battery State of Health (SOH) and Remaining Useful Life (RUL) prediction using the NASA Battery Aging Dataset.

## Project Structure
- data/raw: Raw NASA battery datasets (.mat files)
- data/processed: Processed CSV datasets with extracted cycles, SOH, and RUL
- preprocessing: Data extraction, inspection, SOH calculation, and RUL labeling scripts
- models: Model architectures for health prediction
- 	raining: Model training and optimization pipelines
- evaluation: Model evaluation and performance metrics
- esults: Saved predictions and performance plots

## Preprocessing Pipeline
1. **Dataset Inspection (preprocessing/inspect_dataset.py)**:
   - Loads NASA .mat files using SciPy.
   - Explores operational profiles: charge, discharge, and impedance cycles.
   - Identifies cycle data fields: Voltage, Current, Temperature, Time, and Capacity.
2. **Discharge Data Extraction (preprocessing/extract_discharge_data.py)**:
   - Extracts discharge cycles for batteries **B0005**, **B0006**, **B0007**, and **B0018**.
   - Aligns measurements and saves to data/processed/discharge_data.csv (185,721 measurement rows).
3. **SOH Calculation (preprocessing/calculate_soh.py)**:
   - Calculates State of Health as:
     \text{SOH} = \frac{\text{Capacity}}{\text{Rated Capacity (2.0 Ah)}}
   - Saves to data/processed/discharge_data_with_soh.csv.
4. **EOL Identification (preprocessing/find_eol.py)**:
   - Defines End-of-Life (EOL) threshold as 70% SOH (1.4 Ah capacity) per the base paper.
   - Identifies first EOL discharge cycle for each battery:
     - **B0005**: Cycle 125 (1.397 Ah, 69.8% SOH)
     - **B0006**: Cycle 109 (1.395 Ah, 69.8% SOH)
     - **B0018**: Cycle 97 (1.397 Ah, 69.8% SOH)
     - **B0007**: Did not reach 70% SOH EOL threshold during experiments.
   - Saves results to data/processed/eol_information.csv.
5. **RUL Labeling (preprocessing/calculate_rul.py)**:
   - Calculates Remaining Useful Life for valid batteries up to EOL:
     \text{RUL} = \text{EOL Discharge Cycle} - \text{Current Discharge Cycle}
   - Generates final training dataset: data/processed/discharge_data_with_soh_rul.csv.
