The Pink Elephant Interpretation
================================

Scenario
--------

An autoencoders can be used to denoise images by compressing
images on a layer of smaller dimension and uncompressing it to
retrieve the original image. The compression layer has usually 
no meaning. What if we impose a constraint to this layer which is 
to be as close as possible to a pink elephant without losing too much
on this original denoising task.

* Step 1: train an auto-encoder and visualize the intermediate layer.
* Step 2: train an auto-encoder and tweak the gradient to ressemble a
  pink elephant.
  
Datasets:

* `101 ObjectCategories <https://github.com/mikeizbicki/datasets/tree/master/image/101_ObjectCategories>`_

Scripts as baselines:

* `07_Deconvolution_PY.py <https://github.com/Microsoft/CNTK/blob/master/Examples/Image/GettingStarted/07_Deconvolution_PY.py>`_
* `07_Deconvolution_Visualizer.py <https://github.com/Microsoft/CNTK/blob/master/Examples/Image/GettingStarted/07_Deconvolution_Visualizer.py>`_
* `Basic Autoencoder for Dimensionality Reduction <https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_105_Basic_Autoencoder_for_Dimensionality_Reduction.ipynb>`_
* `Artistic Style Transfer <https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_205_Artistic_Style_Transfer.ipynb>`_

Model descriptions:

* `Image Auto Encoder using deconvolution and unpooling <https://docs.microsoft.com/en-us/cognitive-toolkit/image-auto-encoder-using-deconvolution-and-unpooling>`_

Custom functions in CNTK:

* `Using user functions for debugging <https://cntk.ai/pythondocs/extend.html#using-user-functions-for-debugging>`_
  
The pink elephant:

.. image:: data/pink_elephant.jpg
    :width: 400

Links:

* `Apprentissage sans labels <http://www.xavierdupre.fr/app/ensae_teaching_cs/helpsphinx3/specials/nolabel.html>`_
