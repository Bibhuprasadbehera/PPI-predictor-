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

## [0.4.0] - 2024-11-21

### Changed

-   Modified `data_loader.py` to handle two separate DSSP files (protein 1 and protein 2) and integrate them with the distance matrix.
-   Added functions `load_dssp` and `load_distance_matrix` to handle loading DSSP and distance matrix data, respectively.
-   Created `map_dssp_to_distance_matrix` to generate an enhanced distance matrix incorporating DSSP features (numerical, physicochemical, RSA, SS) for interacting amino acid pairs.
-   Modified `__getitem__` of `ProteinDataset` to return DSSP data for both proteins, the enhanced distance matrix, and the binary interaction matrix.
-   Updated `get_data_loader` to use the modified `ProteinDataset`.
-   Retained and potentially modified `visualize_batch` to handle the new data format.
-   Modified `src/model.py` to accept two separate DSSP inputs and the enhanced distance matrix.
-   Added two separate branches to process DSSP data for each protein using embedding, LSTM, and attention layers.
-   Added convolutional layers to process the enhanced distance matrix.
-   Modified the `forward` method to handle the new inputs and processing steps.
-   Concatenated the processed features from the DSSP branches and the distance matrix branch.
-   Adjusted the fully connected layers to handle the concatenated features.
-   Modified the output layer to produce a matrix of interaction scores.
-   Updated the `__str__` method to reflect the changes in the model architecture.
-   Modified `src/train.py` to use the updated `get_data_loader` function.
-   Removed `ProteinDataset`, `random_split`, and `collate_fn` as they are no longer needed.
-   Updated model instantiation to match the new `ProteinInteractionModel`'s parameters.
-   Modified the training loop to handle the new data format and pass the correct inputs to the model.
-   Added code to move data and model to the correct device (CPU or GPU).
-   Changed the loss function to `MSELoss` and made sure it is appropriate for the new output format.
-   Removed the validation part as the new data loader doesn't provide a separate validation set.
-   Modified `src/evaluate.py` to use the updated `get_data_loader` function.
-   Removed `ProteinDataset` and `collate_fn` as they are no longer needed.
-   Updated model instantiation to match the new `ProteinInteractionModel` parameters.
-   Modified the evaluation loop to handle the new data format and pass the correct inputs to the model.
-   Added code to move data and model to the correct device (CPU or GPU).
-   Removed the call to `visualize_batch`.
-   Updated metric calculation and `create_evaluation_plots` to use the flattened interaction matrix and predictions.
-   Modified `src/predict.py` to work with the new data loader and model.
-   Removed `ProteinDataset` import and usage.
-   Added imports for `load_dssp`, `load_distance_matrix`, and `map_dssp_to_distance_matrix` from `data_loader.py`.
-   Modified `predict` function to accept two protein sequences and their corresponding DSSP file paths, and the distance matrix file path.
-   Loaded DSSP data for both proteins using `load_dssp`.
-   Loaded the distance matrix using `load_distance_matrix`.
-   Created the enhanced distance matrix using `map_dssp_to_distance_matrix`.
-   Updated model instantiation to match the new `ProteinInteractionModel` parameters.
-   Modified the prediction loop to pass the correct inputs to the model and handle the output matrix.
-   Added code to move data and model to the correct device (CPU or GPU).
-   Modified the plotting part to visualize the interaction matrix.
-   Updated argument parsing to accept the new required arguments.
