The data associated to this paper is available from Zenodo (doi: 10.5281/zenodo.5137433).

The histology and mouse measures were generated as part of the Small et al. 2018 study

> Small et al. "Regulatory variants at KLF14 influence type 2 diabetes risk via a female-specific effect on adipocyte size and body composition". Nature Genetics, 50:572–580, 2018.

The hand traced data set, colour maps, and automatic segmentations were generated as part of project cytometer (https://github.com/MRC-Harwell/cytometer), and used for the paper:

> Casero et al. "Phenotyping of Klf14 mouse white adipose tissue enabled by whole slide segmentation with deep neural networks". bioRxiv, 2021. doi: [10.1101/2021.06.03.444997](https://www.biorxiv.org/content/10.1101/2021.06.03.444997v1.full).

List of directories and files:

## `deepcytometer_pipeline_v8.zip` (60.6 MB)

Weights, colourmaps, etc. necessary to run the pipeline (v8, with mode colour correction). This is the version of the pipeline described in the paper.

There are 10 weight files per convolutional neural network (CNN), corresponding to 10-fold cross-validation

* `klf14_b6ntac_exp_0086_cnn_dmap_model_fold_[0..9].h5`:  
Keras weights for the **EDT CNN** (Histology to Euclidean Distance Transform regression)
* `klf14_b6ntac_exp_0089_cnn_segmentation_correction_overlapping_scaled_contours_model_fold_[0..9].h5`:  
Keras weights for the **Correction CNN** (Segmentation Correction regression)
* `klf14_b6ntac_exp_0091_cnn_contour_after_dmap_model_fold_[0..9].h5`:  
Keras weights for the **Contour CNN** (EDT to Contour detection)
* `klf14_b6ntac_exp_0095_cnn_tissue_classifier_fcn_model_fold_[0..9].h5`:  
Keras weights for the **Tissue CNN** (Pixel-wise tissue classifier)
* `klf14_b6ntac_exp_0094_generate_extra_training_images.pickle`:  
training dataset description
  * **'file_list'**:  
  list of SVG files with hand-traced contours for network training. Each SVG file has a corresponding TIFF file with the histology used for segmentation
  * **'idx_test'**:  
  10 lists with file indices for testing in 10-fold cross-validation
  * **'idx_train'**:  
  10 lists with file indices for training in 10-fold cross-validation
  * **'fold_seed'**:  
  seed number used for the random number generator to assign file indices to folds
* `klf14_b6ntac_exp_0098_filename_area2quantile.npz`:  
quantile colour maps calculated in `klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py` using the whole Klf14 data set with v7 of the pipeline, and used in earlier experiments, including some where v8 of the pipeline was used for segmentation.  
* `klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz`:  
quantile colour maps calculated in `klf14_b6ntac_exp_0106_full_slide_pipeline_v8.py` using the whole Klf14 data set with v8 of the pipeline, and used in later experiments.
* `klf14_training_colour_histogram.npz`:  
statistics from Klf14 histology images to be used in colour correction
  * **'xbins_edge'**, **'xbins'**:  
  edges and centres of the bins used for histogram calculations
  * **'hist_r_q1'**, **'hist_r_q2'**, **'hist_r_q3'**
  * **'hist_g_q1'**, **'hist_g_q2'**, **'hist_g_q3'**
  * **'hist_b_q1'**, **'hist_b_q2'**, **'hist_b_q3'**:  
  density quartiles (Q1, Q2, Q3) for RGB channels for each bin the histogram
  * **'mode_r'**, **'mode_g'**, **'mode_b'**:  
  modes for RGB channels (this corresponds to the most typical background colour in the histology images)
  * **'mean_l'**, **'mean_a'**, **'mean_b'**:  
  mean intensity for L*a*b channels of the image
  * **'std_l'**, **'std_a'**, **'std_b'**:  
  intensity standard deviations for L*a*b channels of the image
* `klf14_exp_0112_training_colour_histogram.npz`:  
other statistics from Klf14 histology images to be used in colour correction
  * **'p'**:  
  vector of quantile values used in ECDF calculations
  * **'val_r_klf14'**, **'val_g_klf14'**, **'val_b_klf14'**:  
  all intensity values for the RGB channels of Klf14 training images that contain at least a white adipocyte
  * **'f_ecdf_to_val_r_klf14'**, **'f_ecdf_to_val_g_klf14'**, **'f_ecdf_to_val_b_klf14'**:  
  linear interpolation function that maps ECDF quantiles to intensity values in the Klf14 training data set. These functions can be used together with intensity->quantile interpolation functions calculated for a new histology image to perform histogram matching colour correction
  * **'mean_klf14'**, **'std_klf14'**:  
  mean and standard deviation of the **'val_r_klf14'**, **'val_g_klf14'**, **'val_b_klf14'** vectors

There are also weight files for the pipeline trained with all the data, instead of the 10-fold cross-validation partition. These were not used for the paper, but could be useful for future experiments
* `klf14_b6ntac_exp_0101_cnn_dmap_model.h5`:  
Keras weights for the **EDT CNN** (Histology to Euclidean Distance Transform regression)
* `klf14_b6ntac_exp_0104_cnn_segmentation_correction_overlapping_scaled_contours_model.h5`:  
Keras weights for the **Correction CNN** (Segmentation Correction regression)
* `klf14_b6ntac_exp_0102_cnn_contour_after_dmap_model.h5`:  
Keras weights for the **Contour CNN** (EDT to Contour detection)
* `klf14_b6ntac_exp_0103_cnn_tissue_classifier_fcn_model.h5`:  
Keras weights for the **Tissue CNN** (Pixel-wise tissue classifier)

## `klf14.7z` (2.3 GB)

Mice metadata, training/testing data sets for the pipeline, intermediate files created during training, and neural network weights for multiple experiments.

* `klf14_b6ntac_meta_info.csv`:  
Klf14 mice metadata
  * **Animal Identifier**, **id:** unique ID for each mouse
  * **ko_parent:** heterozygous parent of origin for the KO allele (father, PAT or mother, MAT)
  * **sex:** female or male
  * **genotype:** wild type (KLF14-KO:WT) or heterozygous (KLF14-KO:Het)
  * **BW:** body weight (g)
  * **SC:** subcutaneous depot weight (g)
  * **gWAT:** gonadal depot weight (g)
  * **Liver:** livel weight (g)
  * **cull_age:** age at time of culling (days)
  * **BW_alive:** body weight measured before culling
  * **BW_alive_date:** age at time of BW_alive measure
  * **mother:** unique ID for mouse's mother
  * **mother_genotype:** mouse's mother genotype
* `klf14_b6ntac_training`:  
Directory with hand-traced segmentations of training histology windows. 131 windows sampled from 20 whole slides, plus hand-traced contours that were used for training DeepCytometer and compute population distributions. These segmentations were used for CNN training, but note that there's a cleaned-up version of these data below, and it was the cleaned-up version that was used for the paper experiments
  * `ndpifile_row_YYYYYY_col_XXXXXX[.tif/.xcf/.svg]`: 
    * **ndpifile:** name of the whole slide file (e.g. `KLF14-B6NTAC 36.1c PAT 98-16 C1 - 2016-02-11 10.45.00`)
    * **row_YYYYYY:** Y-coordinate of the top-left corner of the sampling window, in pixels
    * **col_XXXXXX:** X-coordinate of the top-left corner of the sampling window, in pixels
    * **.tif:** TIFF file with the histology sampling window
    * **.xcf:** Gimp file with the histology and hand-traced contours (the contours were drawn in Gimp)
    * **.svg:** SVG (Scalable Vector Graphics) that contains the hand-traced contours in the XCF file
* `klf14_b6ntac_training_v2`:  
Same as `klf14_b6ntac_training`, but the hand-traced data set was cleaned up to remove small contours of dubious cells, or cells that are fully overlapped by others
* `klf14_b6ntac_training_non_overlap`:
Directory with intermediate images to train the networks. These images are generated by script [`klf14_b6ntac_training_non_overlap`](https://github.com/MRC-Harwell/cytometer/blob/main/scripts/klf14_b6ntac_exp_0077_generate_non_overlap_training_images.py)
* `klf14_b6ntac_training_augmented`:
Directory with intermediate images used to train the networks (using augmentation to reduce overfitting). These images are generated by script [`klf14_b6ntac_exp_0078_generate_augmented_training_images.py`](https://github.com/MRC-Harwell/cytometer/blob/main/scripts/klf14_b6ntac_exp_0078_generate_augmented_training_images.py)
* `klf14_b6ntac_seg`:
Deprecated. Directory to store whole slide coarse segmentations in old experiments (e.g. `klf14_b6ntac_exp_0076_generate_training_images.py`). Of little interest for most users
* `klf14_b6ntac_results`:  
Deprecated. Directory to store miscellanea output from some experiments. Of little interest for most users

## `histology.7z` (29.1 GB)

165 H&E histology whole slides from Hamamatsu scanner (`.ndpi`).

## `aida_data_Klf14_v8_images.7z` (16.9 GB)

Histology images converted to DeepZoom so that they can be visualised with [AIDA](https://github.com/alanaberdeen/AIDA).

To use this, decompress this file and put the resulting `images` directory in your `AIDA/dist/data/` directory.
