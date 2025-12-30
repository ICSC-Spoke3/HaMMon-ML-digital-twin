# Utils overview

The `utils/` folder contains small CLI scripts that together implement the automated inference pipeline for large UAV image datasets. The pipeline resizes input images, runs segmentation inference to generate masks, validates the masks, splits multi-class outputs into single-class binary masks, and resizes masks as needed. It is designed to integrate with the photogrammetry pipeline; together they form a science gateway for large-scale UAV data processing.

## File-by-file summary

- `utils/batch_inference.py` runs batch segmentation inference on an image folder and writes predicted masks to an output folder.
- `utils/preprocess_resize.py` validates and resizes input images to a target height (aspect ratio preserved), serving as the first preprocessing stage.
- `utils/masks_validate.py` checks that mask PNGs contain only allowed pixel values (min/max validation).
- `utils/masks_resize.py` validates and resizes mask PNGs to a target height.
- `utils/masks_binary.py` extracts single-class binary masks from multi-class masks for a given label index.
- `utils/split_by_width.py` groups images into folders based on image width, useful for organizing mixed-resolution datasets.
