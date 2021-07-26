Data published in Zenodo (doi: 10.5281/zenodo.5137433).

List of directories and files:

* `deepcytometer_pipeline_v8`:  
weights, colourmaps, etc. necessary to run the pipeline (v8, with mode colour correction).  
There are 10 weights files per convolutional neural network (CNN), corresponding to 10-fold cross-validation
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
    * `'file_list'`:  
    list of SVG files with hand-traced contours for network training. Each SVG file has a corresponding TIFF file with the histology used for segmentation
    * `'idx_test'`:  
    10 lists with file indices for testing in 10-fold cross-validation
    * `'idx_train'`:  
    10 lists with file indices for training in 10-fold cross-validation
    * `'fold_seed'`:  
    seed number used for the random number generator to assign file indices to folds
  * `klf14_b6ntac_exp_0098_filename_area2quantile.npz`:  
  quantile colour maps calculated in `klf14_b6ntac_exp_0098_full_slide_size_analysis_v7.py` using the whole Klf14 data set with v7 of the pipeline, and used in earlier experiments, including some where v8 of the pipeline was used for segmentation.  
  * `klf14_b6ntac_exp_0106_filename_area2quantile_v8.npz`:  
  quantile colour maps calculated in `klf14_b6ntac_exp_0106_full_slide_pipeline_v8.py` using the whole Klf14 data set with v8 of the pipeline, and used in later experiments.
  * `klf14_training_colour_histogram.npz`:  
  statistics from Klf14 histology images to be used in colour correction
    * `'xbins_edge'`, `'xbins'`:  
    edges and centres of the bins used for histogram calculations
    * `'hist_r_q1'`, `'hist_r_q2'`, `'hist_r_q3'`
    * `'hist_g_q1'`, `'hist_g_q2'`, `'hist_g_q3'`
    * `'hist_b_q1'`, `'hist_b_q2'`, `'hist_b_q3'`:  
    density quartiles (Q1, Q2, Q3) for RGB channels for each bin the histogram
    * `'mode_r'`, `'mode_g'`, `'mode_b'`:  
    modes for RGB channels (this corresponds to the most typical background colour in the histology images)
    * `'mean_l'`, `'mean_a'`, `'mean_b'`:  
    mean intensity for L*a*b channels of the image
    * `'std_l'`, `'std_a'`, `'std_b'`:  
    intensity standard deviations for L*a*b channels of the image
  * `klf14_exp_0112_training_colour_histogram.npz`:  
  other statistics from Klf14 histology images to be used in colour correction
    * `'p'`:  
    vector of quantile values used in ECDF calculations
    * `'val_r_klf14'`, `'val_g_klf14'`, `'val_b_klf14'`:  
    all intensity values for the RGB channels of Klf14 training images that contain at least a white adipocyte
    * `'f_ecdf_to_val_r_klf14'`, `'f_ecdf_to_val_g_klf14'`, `'f_ecdf_to_val_b_klf14'`:  
    linear interpolation function that maps ECDF quantiles to intensity values in the Klf14 training data set. These functions can be used together with intensity->quantile interpolation functions calculated for a new histology image to perform histogram matching colour correction
    * `'mean_klf14'`, `'std_klf14'`:  
    mean and standard deviation of the `'val_r_klf14'`, `'val_g_klf14'`, `'val_b_klf14'` vectors
    
