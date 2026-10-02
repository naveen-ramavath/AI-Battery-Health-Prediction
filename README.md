# AI-Battery-Health-Prediction
Implementation of the base paper for AI-based lithium-ion battery SOH and RUL prediction.

## Project Structure
- data/raw: Raw NASA battery datasets (.mat files)
- data/processed: Processed CSV datasets with extracted cycles, SOH, and RUL
- preprocessing: Data extraction, inspection, SOH calculation, and RUL labeling scripts
- models: Model architectures for health prediction
- 	raining: Model training and optimization pipelines
- evaluation: Model evaluation and performance metrics
- esults: Saved predictions and performance plots

## Dataset Exploration & Cycle Selection
- **Batteries**: NASA Li-ion battery aging datasets (B0005, B0006, B0007, B0018).
- **Cycle Types**: Charge, discharge, and impedance.
- **Selection Rationale**: Discharge cycles uniquely record the battery's discharge Capacity (Ah) down to cutoff voltage, which is strictly required for State of Health (SOH) calculation.
