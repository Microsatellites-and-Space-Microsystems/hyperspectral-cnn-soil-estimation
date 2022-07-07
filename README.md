# hyperspectral-cnn-soil-estimation

A simple and light CNN-based regression model for soil parameters estimation from hyperspectral images.
The model has been developed by Achille Ballabeni and Alessandro Lotti[^1] as part of our participation to the ESA’s sponsored <a href="https://platform.ai4eo.eu/seeing-beyond-the-visible">#HYPERVIEW Challenge: Seeing Beyond the Visible</a>.

Our solution ranked 8th over 47 participants.

<b>Overview</b>

The CNN is developed in TensorFlow Keras (2.8.1) and is based on an EfficientNet-Lite model [1-2].
-	train_and_inference.ipynb: Colab Notebook to reproduce our methods
-	toTFRecords: Colab Notebooks to convert the datasets to TFRecords
-	best_model: trained model

[^1]: Member of the <a href="https://site.unibo.it/almasat-lab/en">u3S Laboratory</a> at Alma Mater Studiorum Università di Bologna.

# References

[1] Renjie Liu, <a href="https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html">"Higher accuracy on vision models with EfficientNet-Lite"</a>, TensorFlow Blog.

[2] Sebastian Szymanski, <a href="https://github.com/sebastian-sz/efficientnet-lite-keras">efficientnet-lite-keras</a>.
