"""Copyright (c) 2019 AIT Lab, ETH Zurich, Manuel Kaufmann, Emre Aksan

Students and holders of copies of this code, accompanying datasets,
and documentation, are not allowed to copy, distribute or modify
any of the mentioned materials beyond the scope and duration of the
Machine Perception course projects.

That is, no partial/full copy nor modification of this code and
accompanying data should be made publicly or privately available to
current/future students or other parties.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import zipfile
from constants import Constants as C


def get_activation_fn(activation=C.RELU):
    """
    Return tensorflow activation function given string name.

    Args:
        activation: The requested activation function.

    Returns: The tf op corresponding to the requested activation function.
    """
    # Check if the activation is already callable.
    if callable(activation):
        return activation

    if activation is None:
        return None
    elif activation == C.RELU:
        return tf.nn.relu
    else:
        raise Exception("Activation function is not implemented.")


def export_code(file_list, output_file):
    """
    Adds the given file paths to a zip file.
    Args:
        file_list: List of paths to files
        output_file: Name and path of the zip archive to be created
    """
    zipf = zipfile.ZipFile(output_file, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()


def export_results(eval_result, output_file):
    """
    Write predictions into a csv file that can be uploaded to the submission system.
    Args:
        eval_result: A dictionary {sample_id => (prediction, seed)}. This is exactly what is returned
          by `evaluate_test.evaluate_model`.
        output_file: Where to store the file.
    """

    def to_csv(fname, poses, ids, split=None):
        n_samples, seq_length, dof = poses.shape
        data_r = np.reshape(poses, [n_samples, seq_length * dof])
        cols = ['dof{}'.format(i) for i in range(seq_length * dof)]

        # add split id very last
        if split is not None:
            data_r = np.concatenate([data_r, split[..., np.newaxis]], axis=-1)
            cols.append("split")

        data_frame = pd.DataFrame(data_r,
                                  index=ids,
                                  columns=cols)
        data_frame.index.name = 'Id'

        if not fname.endswith('.gz'):
            fname += '.gz'

        data_frame.to_csv(fname, float_format='%.8f', compression='gzip')

    sample_file_ids = []
    sample_poses = []
    for k in eval_result:
        sample_file_ids.append(k)
        sample_poses.append(eval_result[k][0])

    to_csv(output_file, np.stack(sample_poses), sample_file_ids)
