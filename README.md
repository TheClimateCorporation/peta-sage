# peta-sage
MNIST with Petastorm on Sagemaker using Tensorflow Estimators

* `Build_Verify_Input.ipynb` contains logic to download MNIST data locally and build a petastorm-compatible parquet file. Some code to verify the input & play around with batches and shards is included as well.
* `Invoke_Model.ipynb` sets up sagemaker session and invokes training, once on a single instances and once on a cluster of 5 instances
* `MNIST_model.py` has all the Tensorflow Estimator boilerplate, include the actual DNN model

Code is based on
* [Petastorm MNIST example](https://github.com/uber/petastorm/tree/master/examples/mnist)
* [MNIST DNN solution from Kaggle](https://www.kaggle.com/ilufei/mnist-with-tensorflow-dnn-97)

