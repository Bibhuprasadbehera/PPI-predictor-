# Changelog

## [0.2.0] - 2023-11-20

### Added

- Support for incorporating chain information in the training process.
- Modified `data_loader.py` to load chain IDs from DSSP files.
- Added chain embedding layer to `model.py`.
- Updated `train.py` to use the modified data loader and model.

### Changed

- Updated `data_loader.py`, `plots_dataloader.py`, and `predict.py` to handle chain information correctly.
- Fixed the evaluation and prediction scripts to work with the updated data format.
