# Changelog

## [0.1.0] - 2024-11-14

### Added

- Support for incorporating chain information in the training process.
- Modified `data_loader.py` to load chain IDs from DSSP files.
- Added chain embedding layer to `model.py`.
- Updated `train.py` to use the modified data loader and model.

## [0.2.0] - 2024-11-16

### Changed

- Updated `data_loader.py`, `plots_dataloader.py`, and `predict.py` to handle chain information correctly.
- Fixed the evaluation and prediction scripts to work with the updated data format.

## [0.3.0] - 2024-11-18

### Added

- Utility functions for logging and metric calculation in `utils.py`.
- Integrated logging into `train.py`.
- Integrated metric calculation into `evaluate.py`.

## [0.3.1] - 2024-11-19

### Changed

- Improved error handling in `data_loader.py` for file opening and data type mismatches.
