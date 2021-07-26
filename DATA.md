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
