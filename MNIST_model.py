###
# Adapted from https://github.com/uber/petastorm/blob/master/examples/mnist/tf_example.py
# Model from https://www.kaggle.com/ilufei/mnist-with-tensorflow-dnn-97
###

import argparse
import os
import json
import ast

import tensorflow as tf

from examples.mnist import DEFAULT_MNIST_DATA_PATH
from petastorm import make_reader
from petastorm.tf_utils import make_petastorm_dataset

import logging as _logging
from tensorflow.python.platform import tf_logging

def model_fn(features, labels, mode, params):
    
    # model taken from https://www.kaggle.com/ilufei/mnist-with-tensorflow-dnn-97
    layer1 = tf.keras.layers.Dense(256, activation='relu', input_shape=(params["batch_size"], 784),
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())(features["image"])
    dropped_out = tf.layers.dropout(inputs=layer1, rate=0.4,
                                    training=(mode == tf.estimator.ModeKeys.TRAIN))
    layer2 = tf.keras.layers.Dense(128, activation='relu', 
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())(dropped_out)
    layer3 = tf.keras.layers.Dense(64, activation='relu', 
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())(layer2)
    layer4 = tf.keras.layers.Dense(32, activation='relu', 
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())(layer3)
    layer5 = tf.keras.layers.Dense(16, activation='relu', 
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())(layer4)
    logits = tf.keras.layers.Dense(10,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())(layer5)
    predictions = tf.argmax(logits, 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"preds": predictions},
            export_outputs={'SIGNATURE_NAME': tf.estimator.export.PredictOutput({"preds": predictions})})

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss=cross_entropy, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_entropy, train_op=train_op)
 
    
    accuracy = tf.metrics.accuracy(labels=tf.cast(labels, tf.int64),
                                   predictions=tf.cast(predictions, tf.int64))
    eval_metric_ops = {
        "accuracy": accuracy
    }

    # Provide an estimator spec for `ModeKeys.EVAL` mode.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=cross_entropy,
        eval_metric_ops=eval_metric_ops)


def streaming_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    
    # 28 x 28 is size of MNIST example
    image = tf.cast(tf.reshape(serialized_example.image, [28 * 28]), tf.float32)
    label = serialized_example.digit
 
    return {"image": image}, label
    
    
def _input_fn(reader, batch_size, num_parallel_batches):
    dataset = (make_petastorm_dataset(reader)
               # Per Petastorm docs, do not add a .repeat(num_epochs) here
               # Petastorm will cycle indefinitely through the data given `num_epochs=None`
               # provided to make_reader
               .apply(tf.contrib.data.map_and_batch(streaming_parser,
                                                    batch_size=batch_size,
                                                    num_parallel_batches=num_parallel_batches)))
    return dataset

def main():
    parser = argparse.ArgumentParser(description='Petastorm/Sagemaker/Tensorflow MNIST Example')

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    parser.add_argument('--dataset-url', type=str,
                        metavar='S',
                        help='S3:// URL to the MNIST petastorm dataset')
    
    parser.add_argument('--training_steps', type=int, default=300)
    parser.add_argument('--evaluation_steps', type=int, default=10)
    parser.add_argument('--log_step_count_steps', type=int, default=100)
    parser.add_argument('--save_checkpoints_steps', type=int, default=500)
    parser.add_argument('--save_summary_steps', type=int, default=50)
    parser.add_argument('--throttle_secs', type=int, default=10)
    
    parser.add_argument('--prefetch_size', type=int, default=16)
    parser.add_argument('--num_parallel_batches', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    
    args = parser.parse_args()
        
    tf.logging.set_verbosity(tf.logging.DEBUG)
 
    # TF 1.13 and 1.14 handle logging a bit different, so wrapping the logging setup in a try/except block
    try:
        tf_logger = tf_logging._get_logger()
        handler = tf_logger.handlers[0]
        handler.setFormatter(_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    except:
        pass
    
    # In 1.14, a multi-worker synchronous training can be achieved using CollectiveAllReduceStrategy per
    # See https://github.com/tensorflow/tensorflow/issues/23664
    # Without providing train_distribute, I believe asynchronous training is done
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=args.save_checkpoints_steps,
                                        log_step_count_steps=args.log_step_count_steps,
                                        save_summary_steps=args.save_summary_steps,)
    
    model_dir_parent_path = args.model_dir[:-5]
    model_dir_parent = model_dir_parent_path.split("/")[-2]
    
    print(f"Launch tensorboard by running the following in terminal:\n" +
          "aws s3 sync {model_dir_parent_path} ~/Downloads/{model_dir_parent} && " +
          "tensorboard --logdir=~/Downloads/{model_dir_parent}")
    
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params={"batch_size": args.batch_size},
        config=run_config)

    workers = json.loads(os.environ['SM_HOSTS'])
    worker_index = workers.index(os.environ['SM_CURRENT_HOST'])
    nr_workers = len(workers)
    print(f"Inside training script on worker with (0-based) index {worker_index} out of {nr_workers - 1}.")

    with make_reader(os.path.join(args.dataset_url, 'train'),
                     num_epochs=None,
                     cur_shard=worker_index,
                     shard_count=nr_workers,
                     workers_count=nr_workers) as train_reader:
        with make_reader(os.path.join(args.dataset_url, 'test'),
                         num_epochs=None,
                         cur_shard=0,
                         shard_count=1) as eval_reader:
            
            train_fn = lambda : _input_fn(reader = train_reader,
                                          batch_size=args.batch_size,
                                          num_parallel_batches=args.num_parallel_batches)

            eval_fn = lambda : _input_fn(reader = eval_reader,
                                         batch_size=args.batch_size,
                                         num_parallel_batches=args.num_parallel_batches)

            train_spec = tf.estimator.TrainSpec(input_fn=train_fn,
                                                max_steps=args.training_steps)

            eval_spec = tf.estimator.EvalSpec(input_fn=eval_fn,
                                              throttle_secs=args.throttle_secs,
                                              steps=args.evaluation_steps)

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    main()
