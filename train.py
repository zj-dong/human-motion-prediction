"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
import os
import time
import argparse
import json
import numpy as np
import tensorflow as tf

import tf_models as models
from tf_data import TFRecordMotionDataset
from constants import Constants as C
from motion_metrics import MetricsEngine

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data_dir', required=True, default='./data', help='Where the data (tfrecords) is stored.')
parser.add_argument('--save_dir', required=True, default='./experiments', help='Where to save checkpoints to.')
parser.add_argument("--seq_length_in", type=int, default=120, help="Number of input frames (60 fps).")
parser.add_argument("--seq_length_out", type=int, default=24, help="Number of output frames (60 fps).")

# Learning
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate.')
parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use during training.")

# Architecture
parser.add_argument("--model_type", type=str, default="dummy", help="Model to train.")
parser.add_argument("--cell_type", type=str, default="lstm", help="RNN cell type: lstm, gru")
parser.add_argument("--cell_size", type=int, default=1024, help="RNN cell size.")
parser.add_argument("--num_layers", type=int, default=3, help="number of rnns to stack.")
parser.add_argument("--input_hidden_size", type=int, default=None, help="Input dense layer before the recurrent cell.")
parser.add_argument("--activation_fn", type=str, default=None, help="Activation Function on the output.")
parser.add_argument("--loss_to_use",type=str,default="sampling_based")
parser.add_argument("--architecture",type=str,default="basic",help="basic or tied")

# Training
parser.add_argument("--num_epochs", type=int, default=400, help="Number of training epochs.")
parser.add_argument("--print_every", type=int, default=200, help="How often to log training error.")
parser.add_argument("--test_every", type=int, default=500, help="How often to compute the error on the validation set.")
parser.add_argument("--use_cpu", action="store_true", help="Use CPU instead of GPU.")
parser.add_argument("--experiment_name", type=str, default=None, help="A descriptive name for the experiment.")


ARGS = parser.parse_args()
EXPERIMENT_TIMESTAMP = str(int(time.time()))


def create_model(session):
    # Global step variable.
    global_step = tf.Variable(1, trainable=False, name='global_step')

    # Get the paths to the TFRecord files.
    data_path = ARGS.data_dir
    train_data_path = os.path.join(data_path, "training", "poses-?????-of-?????")
    valid_data_path = os.path.join(data_path, "validation", "poses-?????-of-?????")
    meta_data_path = os.path.join(data_path, "training", "stats.npz")
    train_dir = ARGS.save_dir

    # Parse the commandline arguments to a more readable config.
    if ARGS.model_type == "dummy":
        model_cls, config, experiment_name = get_dummy_config(ARGS)
    else:
        raise Exception("Model type '{}' unknown.".format(ARGS.model_type))

    # Create a folder for the experiment.
    experiment_dir = os.path.normpath(os.path.join(train_dir, experiment_name))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # Load the training data.
    window_length = ARGS.seq_length_in + ARGS.seq_length_out
    with tf.name_scope("training_data"):
        train_data = TFRecordMotionDataset(data_path=train_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=ARGS.batch_size,
                                           shuffle=True,
                                           extract_windows_of=window_length,
                                           extract_random_windows=True,
                                           num_parallel_calls=16)
        train_pl = train_data.get_tf_samples()

    # Load validation data.
    with tf.name_scope("validation_data"):
        valid_data = TFRecordMotionDataset(data_path=valid_data_path,
                                           meta_data_path=meta_data_path,
                                           batch_size=ARGS.batch_size,
                                           shuffle=False,
                                           extract_windows_of=window_length,
                                           extract_random_windows=False,
                                           num_parallel_calls=16)
        valid_pl = valid_data.get_tf_samples()

    # Create the training model.
    with tf.name_scope(C.TRAIN):
        train_model = model_cls(
            config=config,
            data_pl=train_pl,
            mode=C.TRAIN,
            reuse=False,
            dtype=tf.float32)
        train_model.build_graph()

    # Create a copy of the training model for validation.
    with tf.name_scope(C.EVAL):
        valid_model = model_cls(
            config=config,
            data_pl=valid_pl,
            mode=C.EVAL,
            reuse=True,
            dtype=tf.float32)
        valid_model.build_graph()

    # Count and print the number of trainable parameters.
    num_param = 0
    for v in tf.trainable_variables():
        num_param += np.prod(v.shape.as_list())
    print("# of parameters: " + str(num_param))
    config["num_parameters"] = int(num_param)

    # Dump the config to the experiment directory.
    json.dump(config, open(os.path.join(experiment_dir, 'config.json'), 'w'), indent=4, sort_keys=True)
    print("Experiment directory " + experiment_dir)

    # Create the optimizer for the training model.
    train_model.optimization_routines()

    # Create the summaries for tensoboard.
    train_model.summary_routines()
    valid_model.summary_routines()

    # Create the saver object to store checkpoints. We keep track of only 1 checkpoint.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1, save_relative_paths=True)

    # Initialize the variables.
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())

    models = [train_model, valid_model]
    data = [train_data, valid_data]
    return models, data, saver, global_step, experiment_dir


def load_latest_checkpoint(sess, saver, experiment_dir):
    """Restore the latest checkpoint found in `experiment_dir`."""
    ckpt = tf.train.get_checkpoint_state(experiment_dir, latest_filename="checkpoint")

    if ckpt and ckpt.model_checkpoint_path:
        # Check if the specific checkpoint exists
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Loading model checkpoint {0}".format(ckpt_name))
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError("could not load checkpoint")


def get_dummy_config(args):
    """
    Create a config from the parsed commandline arguments that is more readable. You can use this to define more
    parameters and their default values.
    Args:
        args: The parsed commandline arguments.

    Returns:
        The model class, the config, and the experiment name.
    """
    assert args.model_type == "dummy"

    config = dict()
    config['model_type'] = args.model_type
    config['seed'] = C.SEED
    config['learning_rate'] = args.learning_rate
    config['cell_type'] = args.cell_type
    config['num_layers'] = args.num_layers
    config['cell_size'] = args.cell_size
    config['input_hidden_size'] = args.input_hidden_size
    config['source_seq_len'] = args.seq_length_in
    config['target_seq_len'] = args.seq_length_out
    config['batch_size'] = args.batch_size
    config['activation_fn'] = args.activation_fn
    config['loss_to_use']  = args.loss_to_use
    config['architecture'] = args.architecture
    model_cls = models.S2sModel

    # Create an experiment name that summarizes the configuration.
    # It will be used as part of the experiment folder name.
    experiment_name_format = "{}-{}{}-b{}-{}@{}-in{}_out{}"
    experiment_name = experiment_name_format.format(EXPERIMENT_TIMESTAMP,
                                                    args.model_type,
                                                    "-"+args.experiment_name if args.experiment_name is not None else "",
                                                    config['batch_size'],
                                                    config['cell_size'],
                                                    config['cell_type'],
                                                    args.seq_length_in,
                                                    args.seq_length_out)
    return model_cls, config, experiment_name


def train():
    """
    The main training loop. Loads the data, creates the model, and trains for the specified number of epochs.
    """
    # Limit TF to take a fraction of the GPU memory
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    device_count = {"GPU": 0} if ARGS.use_cpu else {"GPU": 1}
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)) as sess:

        # Create the models and load the data.
        models, data, saver, global_step, experiment_dir = create_model(sess)
        train_model, valid_model = models
        train_data, valid_data = data

        # Create metrics engine including summaries
        target_lengths = [x for x in C.METRIC_TARGET_LENGTHS if x <= train_model.target_seq_len]
        metrics_engine = MetricsEngine(target_lengths)
        # create the necessary summary placeholders and ops
        metrics_engine.create_summaries()
        # reset computation of metrics
        metrics_engine.reset()

        # Summary writers for train and test runs
        summaries_dir = os.path.normpath(os.path.join(experiment_dir, "log"))
        train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
        valid_writer = train_writer
        print("Model created")

        # Training loop configuration.
        stop_signal = False
        time_counter = 0.0
        step = 1
        epoch = 0
        train_loss = 0.0
        train_iter = train_data.get_iterator()
        valid_iter = valid_data.get_iterator()

        print("Running Training Loop.")
        # Initialize the data iterators.
        sess.run(train_iter.initializer)
        sess.run(valid_iter.initializer)

        def evaluate_model(_eval_model, _eval_iter, _metrics_engine, _return_results=False):
            # make a full pass on the validation set and compute the metrics
            _eval_result = dict()
            _start_time = time.perf_counter()
            _metrics_engine.reset()
            sess.run(_eval_iter.initializer)
            try:
                while True:
                    # get the predictions and ground truth values
                    #predictions, targets, seed_sequence, data_id = _eval_model.sampled_step(sess)
                    loss, summary,targets, predictions = _eval_model.sampled_step(sess)
                    predictions = np.array(predictions)
                    targets = np.array(targets)
                    predictions = predictions.transpose((1,0,2))
                    #print(predictions.shape)
                    #print(targets.shape)
                    _metrics_engine.compute_and_aggregate(predictions, targets)

                    if _return_results:
                        # Store each test sample and corresponding predictions with the unique sample IDs.
                        for k in range(predictions.shape[0]):
                            _eval_result[data_id[k].decode("utf-8")] = (predictions[k],
                                                                        targets[k],
                                                                        seed_sequence["poses"][k])

            except tf.errors.OutOfRangeError:
                # finalize the computation of the metrics
                final_metrics = _metrics_engine.get_final_metrics()
            return final_metrics, time.perf_counter() - _start_time, _eval_result

        while not stop_signal:
            # Training.
            for i in range(ARGS.test_every):
                try:
                    start_time = time.perf_counter()
                    step += 1

                    step_loss, summary, prediction = train_model.step(sess)
                   
                    train_writer.add_summary(summary, step)
                    train_loss += step_loss

                    time_counter += (time.perf_counter() - start_time)
                    if step % ARGS.print_every == 0:
                        train_loss_avg = train_loss / ARGS.print_every
                        time_elapsed = time_counter / ARGS.print_every
                        train_loss, time_counter = 0., 0.
                        print("Train [{:04d}] \t Loss: {:.3f} \t time/batch: {:.3f}".format(step,
                                                                                            train_loss_avg,
                                                                                            time_elapsed))

                except tf.errors.OutOfRangeError:
                    sess.run(train_iter.initializer)
                    epoch += 1
                    if epoch >= ARGS.num_epochs:
                        stop_signal = True
                        break

            # Evaluation: make a full pass on the validation split.
            valid_metrics, valid_time, _ = evaluate_model(valid_model, valid_iter, metrics_engine)
            # print an informative string to the console
            print("Valid [{:04d}] \t {} \t total_time: {:.3f}".format(step - 1,
                                                                      metrics_engine.get_summary_string(valid_metrics),
                                                                      valid_time))

            # Write summaries to tensorboard.
            summary_feed = metrics_engine.get_summary_feed_dict(valid_metrics)
            summaries = sess.run(metrics_engine.all_summaries_op, feed_dict=summary_feed)
            valid_writer.add_summary(summaries, step)

            # Reset metrics and iterator.
            metrics_engine.reset()
            sess.run(valid_iter.initializer)

            # Save the model. You might want to think about if it's always a good idea to do that.
            print("Saving the model to {}".format(experiment_dir))
            saver.save(sess, os.path.normpath(os.path.join(experiment_dir, 'checkpoint')), global_step=step-1)

        print("End of Training.")

        print("Evaluating validation set ...")
        load_latest_checkpoint(sess, saver, experiment_dir)
        valid_metrics, valid_time, _ = evaluate_model(valid_model, valid_iter, metrics_engine)
        print("Valid [{:04d}] \t {} \t total_time: {:.3f}".format(step - 1,
                                                                  metrics_engine.get_summary_string(valid_metrics),
                                                                  valid_time))

        print("Training Finished.")


if __name__ == "__main__":
    train()
