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

## [0.4.0] - 2024-11-26

### Changed

- Modified the model to accept two sequences as input and predict their interaction score.
- Updated `src/predict.py` to handle two input sequences, generate plots for each, and save the predicted interaction score to a CSV file.
- Updated `src/model.py` to handle two sequences in the forward pass and calculate the interaction score.

## [0.5.0] - 2024-11-26

### Changed

- Modified the model to handle sequences of different lengths during prediction and output an interaction matrix.
- Updated `src/predict.py` to accept two amino acid sequences as input, pad them if necessary, and save the interaction matrix to a CSV file.
- Updated `src/model.py` to handle sequences of different lengths in the forward pass and calculate the interaction matrix.
- Removed RSA, secondary structure, and physicochemical properties as input requirements during prediction.

## [0.6.0] - 2024-11-27

### Changed

- Updated `src/train.py` to handle interaction matrix output during training.
- Modified the data loading process in `src/train.py` to extract sequences and features based on chain IDs.
- Updated the loss function in `src/train.py` to handle the interaction matrix output.

## [0.7.0] - 2024-11-27

### Changed

- Updated `src/model.py` to handle padding correctly during the attention calculation.
- Added masking to the attention mechanism to ignore padding tokens.
