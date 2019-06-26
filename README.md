# peta-sage
MNIST with Petastorm on Sagemaker using Tensorflow Estimators

* `Build_Verify_Input.ipynb` contains logic to download MNIST data locally and build a petastorm-compatible parquet file. Some code to verify the input & play around with batches and shards is included as well.
* `Invoke_Model.ipynb` sets up sagemaker session and invokes training, once on a single instances and once on a cluster of 5 instances
* `MNIST_model.py` has all the Tensorflow Estimator boilerplate, include the actual DNN model

Code is based on
* [Petastorm MNIST example](https://github.com/uber/petastorm/tree/master/examples/mnist)
* [MNIST DNN solution from Kaggle](https://www.kaggle.com/ilufei/mnist-with-tensorflow-dnn-97)

## Sagemaker Docker Container with Sagemaker

In order to run this code in sagemaker, you need a docker container that has the appropriate
petastorm libraries installed.  This container can be built by following the instructions 
[here](https://github.com/aws/sagemaker-tensorflow-container) to build and deploy your modified version of the containers.  The only change necessary is to add the following into docker/{version}/Dockerfile.cpu|gpu right before the final statement.

    RUN pip install petastorm
    RUN pip install s3fs


