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
import tensorflow as tf

from constants import Constants as C
from utils import get_activation_fn
import rnn_cell_extensions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
class BaseModel(object):
    """
    Base class that defines some functions and variables commonly used by all models. Subclass `BaseModel` to
    create your own models (cf. `DummyModel` for an example).
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        self.config = config  # The config parameters from the train.py script.
        self.data_placeholders = data_pl  # Placeholders where the input data is stored.
        self.mode = mode  # Train or eval.
        self.reuse = reuse  # If we want to reuse existing weights or not.
        self.source_seq_len = config["source_seq_len"]  # Length of the input seed.
        self.target_seq_len = config["target_seq_len"]  # Length of the predictions to be made.
        self.batch_size = config["batch_size"]  # Batch size.
        self.activation_fn_out = get_activation_fn(config["activation_fn"])  # Output activation function.
        self.data_inputs = data_pl[C.BATCH_INPUT]  # Tensor of shape (batch_size, seed length + target length)
        self.data_targets = data_pl[C.BATCH_TARGET]  # Tensor of shape (batch_size, seed length + target length)
        self.data_seq_len = data_pl[C.BATCH_SEQ_LEN]  # Tensor of shape (batch_size, )
        self.data_ids = data_pl[C.BATCH_ID]  # Tensor of shape (batch_size, )
        self.is_eval = self.mode == C.EVAL  # If we are in evaluation mode.
        self.is_training = self.mode == C.TRAIN  # If we are in training mode.
        self.is_test = self.mode == C.TEST
        self.global_step = tf.train.get_global_step(graph=None)  # Stores the number of training iterations.

        # The following members should be set by the child class.
        self.outputs = None  # The final predictions.
        self.outputs_prev = None 
        self.prediction_targets = None  # The targets.
        self.prediction_inputs = None  # The inputs used to make predictions.
        self.prediction_representation = None  # Intermediate representations.
        self.loss = None  # Loss op to be used during training.
        self.learning_rate = config["learning_rate"]  # Learning rate.
        self.parameter_update = None  # The training op.
        self.summary_update = None  # Summary op.

        # Hard-coded parameters that define the input size.
        self.JOINT_SIZE = 3*3
        self.NUM_JOINTS = 15
        self.HUMAN_SIZE = self.NUM_JOINTS*self.JOINT_SIZE
        self.input_size = self.HUMAN_SIZE

    def build_graph(self):
        """Build this model, i.e. its computational graph."""
        self.build_network()

    def build_network(self):
        """Build the core part of the model. This must be implemented by the child class."""
        raise NotImplementedError()

    def build_loss(self):
        """Build the loss function."""
        if self.is_eval:
            # In evaluation mode (for the validation set) we only want to know the loss on the target sequence,
            # because the seed sequence was just used to warm up the model.
            predictions_pose = self.outputs[:, -self.target_seq_len:, :]
            targets_pose = self.prediction_targets[:, -self.target_seq_len:, :]
        else:
            predictions_pose = self.outputs
            targets_pose = self.prediction_targets

        # Use MSE loss.
        with tf.name_scope("loss"):
            diff = targets_pose - predictions_pose
            self.loss = tf.reduce_mean(tf.square(diff))

    def optimization_routines(self):
        """Add an optimizer."""
        # Use a simple SGD optimizer.
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Gradients and update operation for training the model.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            # In case you want to do anything to the gradients, here you could do it.
            self.parameter_update = optimizer.apply_gradients(grads_and_vars=zip(gradients, params),
                                                              global_step=self.global_step)

    def build_output_layer(self):
        """Build the final dense output layer without any activation."""
        with tf.variable_scope("output_layer", reuse=self.reuse):
            self.outputs = tf.layers.dense(self.prediction_representation, self.input_size,
                                           self.activation_fn_out, reuse=self.reuse)

    def summary_routines(self):
        """Create the summary operations necessary to write logs into tensorboard."""
        # Note that summary_routines are called outside of the self.mode name_scope. Hence, self.mode should be
        # prepended to the summary name if needed.
        tf.summary.scalar(self.mode+"/loss", self.loss, collections=[self.mode+"/model_summary"])

        if self.is_training:
            tf.summary.scalar(self.mode + "/learning_rate",
                              self.learning_rate,
                              collections=[self.mode + "/model_summary"])

        self.summary_update = tf.summary.merge_all(self.mode+"/model_summary")

    def step(self, session):
        """
        Perform one training step, i.e. compute the predictions when we can assume ground-truth is available.
        """
        raise NotImplementedError()

    def sampled_step(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This still assumes
        that we have ground-truth available."""
        raise NotImplementedError()

    def predict(self, session):
        """
        Compute the predictions given the seed sequence without having access to the ground-truth values.
        """
        raise NotImplementedError()


class DummyModel(BaseModel):
    """
    A dummy RNN model.
    """
    def __init__(self, config, data_pl, mode, reuse, **kwargs):
        super(DummyModel, self).__init__(config, data_pl, mode, reuse, **kwargs)

        # Extract some config parameters specific to this model
        self.cell_type = self.config["cell_type"]
        self.cell_size = self.config["cell_size"]
        self.input_hidden_size = self.config.get("input_hidden_size")
        self.loss_to_use

        # Prepare some members that need to be set when creating the graph.
        self.cell = None  # The recurrent cell.
        self.initial_states = None  # The intial states of the RNN.
        self.rnn_outputs = None  # The outputs of the RNN layer.
        self.rnn_state = None  # The final state of the RNN layer.
        self.inputs_hidden = None  # The inputs to the recurrent cell.

        # How many steps we must predict.
        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1 #143
        else:
            self.sequence_length = self.target_seq_len

        self.prediction_inputs = self.data_inputs[:, :-1, :]  # Pose input.
        self.prediction_targets = self.data_inputs[:, 1:, :]  # The target poses for every time step.
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length

        # Sometimes the batch size is available at compile time.
        self.tf_batch_size = self.prediction_inputs.shape.as_list()[0]
        if self.tf_batch_size is None:
            # Sometimes it isn't. Use the dynamic shape instead.
            self.tf_batch_size = tf.shape(self.prediction_inputs)[0]

    def build_input_layer(self):
        """
        Here we can do some stuff on the inputs before passing them to the recurrent cell. The processed inputs should
        be stored in `self.inputs_hidden`.
        """
        # We could e.g. pass them through a dense layer
        if self.input_hidden_size is not None:
            with tf.variable_scope("input_layer", reuse=self.reuse):
                self.inputs_hidden = tf.layers.dense(self.prediction_inputs, self.input_hidden_size,
                                                     tf.nn.relu, self.reuse)
        else:
            self.inputs_hidden = self.prediction_inputs

    def build_cell(self):
        """Create recurrent cell."""
        with tf.variable_scope("rnn_cell", reuse=self.reuse):
            if self.cell_type == C.LSTM:
                cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
            elif self.cell_type == C.GRU:
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
            else:
                raise ValueError("Cell type '{}' unknown".format(self.cell_type))

            self.cell = cell

    def build_network(self):
        """Build the core part of the model."""
        self.build_input_layer()
        self.build_cell()

        self.initial_states = self.cell.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)
        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                                 self.inputs_hidden,
                                                                 sequence_length=self.prediction_seq_len,
                                                                 initial_state=self.initial_states,
                                                                 dtype=tf.float32)
            self.prediction_representation = self.rnn_outputs
        self.build_output_layer()
        self.build_loss()

    def build_loss(self):
        super(DummyModel, self).build_loss()

    def step(self, session):
        """
        Run a training or validation step of the model.
        Args:
          session: Tensorflow session object.
        Returns:
          A triplet of loss, summary update and predictions.
        """
        if self.is_training:
            # Training step.
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step (no backprop).
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This still assumes
        that we have ground-truth available.
        Args:
          session: Tensorflow session object.
        Returns:
          Prediction with shape (batch_size, self.target_seq_len, feature_size), ground-truth targets, seed sequence and
          unique sample IDs.
        """
        assert self.is_eval, "Only works in evaluation mode."

        # Get the current batch.
        batch = session.run(self.data_placeholders)
        data_id = batch[C.BATCH_ID]
        data_sample = batch[C.BATCH_INPUT]
        targets = data_sample[:, self.source_seq_len:]

        seed_sequence = data_sample[:, :self.source_seq_len]
        predictions = self.sample(session, seed_sequence, prediction_steps=self.target_seq_len)

        return predictions, targets, seed_sequence, data_id

    def predict(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This assumes no
        ground-truth data is available.
        Args:
            session: Tensorflow session object.

        Returns:
            Prediction with shape (batch_size, self.target_seq_len, feature_size), seed sequence and unique sample IDs.
        """
        # `sampled_step` is written such that it works when no ground-truth data is available, too.
        predictions, _, seed, data_id = self.sampled_step(session)
        return predictions, seed, data_id

    def sample(self, session, seed_sequence, prediction_steps):
        """
        Generates `prediction_steps` may poses given a seed sequence.
        Args:
            session: Tensorflow session object.
            seed_sequence: A tensor of shape (batch_size, seq_len, feature_size)
            prediction_steps: How many frames to predict into the future.
            **kwargs:
        Returns:
            Prediction with shape (batch_size, prediction_steps, feature_size)
        """
        assert self.is_eval, "Only works in sampling mode."
        one_step_seq_len = np.ones(seed_sequence.shape[0])

        # Feed the seed sequence to warm up the RNN.
        feed_dict = {self.prediction_inputs: seed_sequence,
                     self.prediction_seq_len: np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]}
        state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)

        # Now create predictions step-by-step.
        prediction = prediction[:, -1:]
        predictions = [prediction]
        for step in range(prediction_steps-1):
            # get the prediction
            feed_dict = {self.prediction_inputs: prediction,
                         self.initial_states: state,
                         self.prediction_seq_len: one_step_seq_len}
            state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)
            predictions.append(prediction)
        return np.concatenate(predictions, axis=1)


class S2sModel(BaseModel):
    """
    A dummy RNN model.
    """
    def __init__(self, config, data_pl, mode, reuse,**kwargs):
        super(S2sModel, self).__init__(config, data_pl, mode, reuse, **kwargs)

        # Extract some config parameters specific to this model
        self.cell_type = self.config["cell_type"]
        self.cell_size = self.config["cell_size"]
        self.num_layers  = self.config["num_layers"]
        self.input_hidden_size = self.config.get("input_hidden_size")
        self.loss_to_use = self.config["loss_to_use"]
        self.architecture = self.config["architecture"]
        
        # Prepare some members that need to be set when creating the graph.
        self.cell = None  # The recurrent cell.
        self.initial_states = None  # The intial states of the RNN.
        self.rnn_outputs = None  # The outputs of the RNN layer.
        self.rnn_state = None  # The final state of the RNN layer.
        self.inputs_hidden = None  # The inputs to the recurrent cell.

        # How many steps we must predict.
        if self.is_training:
            self.sequence_length = self.source_seq_len + self.target_seq_len - 1 #143
        else:
            self.sequence_length = self.target_seq_len

        self.prediction_inputs = self.data_inputs[:, :self.source_seq_len, :]  # Pose input.
        self.prediction_targets = self.data_inputs[:, 1:, :]  # The target poses for every time step.
        self.prediction_seq_len = tf.ones((tf.shape(self.prediction_targets)[0]), dtype=tf.int32)*self.sequence_length
        
        if self.is_test:
            self.encoder_inputs = self.data_inputs[:, :self.source_seq_len-1, :]
            var = tf.ones([tf.shape(self.encoder_inputs)[0],self.target_seq_len-1,self.input_size])
            self.decoder_inputs = tf.concat([self.data_inputs[:, self.source_seq_len-1:, :],var],1)
            self.decoder_outputs = self.data_inputs[:, self.source_seq_len:, 0:self.input_size]
            self.encoder_prev = self.data_inputs[:, :self.source_seq_len, :]
            
        else:
            self.encoder_inputs = self.data_inputs[:, :self.source_seq_len-1, :]
            self.decoder_inputs = self.data_inputs[:, self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
            self.decoder_outputs = self.data_inputs[:, self.source_seq_len:, 0:self.input_size]
            self.encoder_prev = self.data_inputs[:, :self.source_seq_len, :]
            
        encoder_inputs=[]
        decoder_inputs=[]
        decoder_outputs=[]
        encoder_prev = []
        encoder_inputs.append(self.encoder_inputs)
        decoder_inputs.append(self.decoder_inputs)
        decoder_outputs.append(self.decoder_outputs)
        encoder_prev.append(self.encoder_prev)
        #self.encoder_inputs = enc_in
        #self.decoder_inputs = dec_in
        #self.decoder_outputs = dec_out
        
        encoder_inputs[-1] = tf.reshape(tf.transpose(encoder_inputs[-1], [1, 0, 2]),[-1, self.input_size])
        decoder_inputs[-1] = tf.reshape(tf.transpose(decoder_inputs[-1], [1, 0, 2]),[-1, self.input_size])
        decoder_outputs[-1]= tf.reshape(tf.transpose(decoder_outputs[-1], [1, 0, 2]),[-1, self.input_size])
        encoder_prev[-1]= tf.reshape(tf.transpose(encoder_prev[-1], [1, 0, 2]),[-1, self.input_size])
        
        #self.encoder_inputs = tf.reshape(self.encoder_inputs, [-1, self.input_size])
        #self.decoder_inputs = tf.reshape(self.decoder_inputs, [-1, self.input_size])
        #self.decoder_outputs = tf.reshape(self.decoder_outputs, [-1, self.input_size])
        if self.is_test:

            self.enc_in = tf.split(encoder_inputs[-1] , self.source_seq_len-1, axis=0)
            self.dec_in = tf.split(decoder_inputs[-1],self.target_seq_len, axis=0)
            self.dec_out = tf.split(decoder_outputs[-1], self.target_seq_len, axis=0)
            self.enc_prev = tf.split(encoder_prev[-1] , self.source_seq_len, axis=0)
        else:
            self.enc_in = tf.split(encoder_inputs[-1] , self.source_seq_len-1, axis=0)
            self.dec_in = tf.split(decoder_inputs[-1],self.target_seq_len, axis=0)
            self.dec_out = tf.split(decoder_outputs[-1], self.target_seq_len, axis=0)
            self.enc_prev = tf.split(encoder_prev[-1] , self.source_seq_len, axis=0)
        
        '''
               
        enc_in = tf.placeholder(dtype, shape=[None, source_seq_len-1, self.input_size], name="enc_in")
      dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_in")
      dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_out")

      self.encoder_inputs = enc_in
      self.decoder_inputs = dec_in
      self.decoder_outputs = dec_out

      enc_in = tf.transpose(enc_in, [1, 0, 2])
      dec_in = tf.transpose(dec_in, [1, 0, 2])
      dec_out = tf.transpose(dec_out, [1, 0, 2])

      enc_in = tf.reshape(enc_in, [-1, self.input_size])
      dec_in = tf.reshape(dec_in, [-1, self.input_size])
      dec_out = tf.reshape(dec_out, [-1, self.input_size])

      enc_in = tf.split(enc_in, source_seq_len-1, axis=0)
      dec_in = tf.split(dec_in, target_seq_len, axis=0)
      dec_out = tf.split(dec_out, target_seq_len, axis=0)
      '''
        ''' self.enc_in = self.data_inputs[:, :self.source_seq_len-1, :]
        self.dec_in = self.data_inputs[:, self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
        self.dec_out = self.data_inputs[:, self.source_seq_len:, 0:self.input_size]'''
        # Sometimes the batch size is available at compile time.
        self.tf_batch_size = self.prediction_inputs.shape.as_list()[0]
        if self.tf_batch_size is None:
            # Sometimes it isn't. Use the dynamic shape instead.
            self.tf_batch_size = tf.shape(self.prediction_inputs)[0]
        
            
            
           
    def build_input_layer(self):
        """
        Here we can do some stuff on the inputs before passing them to the recurrent cell. The processed inputs should
        be stored in `self.inputs_hidden`.
        """
        # We could e.g. pass them through a dense layer
        if self.input_hidden_size is not None:
            with tf.variable_scope("input_layer", reuse=self.reuse):
                self.inputs_hidden = tf.layers.dense(self.prediction_inputs, self.input_hidden_size,
                                                     tf.nn.relu, self.reuse)
        else:
            self.inputs_hidden = self.prediction_inputs

    def build_cell(self,residual_velocities=True):
        """Create recurrent cell."""
        cell = tf.contrib.rnn.LSTMCell(self.cell_size,reuse=self.reuse,initializer=tf.orthogonal_initializer(),forget_bias=0.2)
        cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=64, state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-0.5)
        #cell = tf.contrib.rnn.BidirectionalGridLSTMCell(self.cell_size,reuse=self.reuse,initializer=tf.orthogonal_initializer(),forget_bias=0.2)
        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell( [tf.contrib.rnn.LSTMCell(1024,reuse=self.reuse,initializer=tf.orthogonal_initializer(),forget_bias=0.2) for _ in range(self.num_layers)] )
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1-0.5)
        cell = rnn_cell_extensions.LinearSpaceDecoderWrapper( cell, self.input_size,reuse=self.reuse)
        if residual_velocities:
            cell = rnn_cell_extensions.ResidualWrapper(cell)
        
        
        '''
        with tf.variable_scope("rnn_cell", reuse=self.reuse):
            if self.cell_type == C.LSTM:
                cell = tf.nn.rnn_cell.LSTMCell(self.cell_size, reuse=self.reuse)
            elif self.cell_type == C.GRU:
                cell = tf.nn.rnn_cell.GRUCell(self.cell_size, reuse=self.reuse)
            else:
                raise ValueError("Cell type '{}' unknown".format(self.cell_type))
        '''
        self.cell = cell

    def build_network(self):
        """Build the core part of the model."""
        #self.build_input_layer()
        self.build_cell()

        #self.initial_states = self.cell.zero_state(batch_size=self.tf_batch_size, dtype=tf.float32)
        lf = None
        if self.loss_to_use == "sampling_based":
            def lf(prev, i): # function for sampling_based loss
                return prev
        elif self.loss_to_use == "supervised":
            pass
        else:
            raise(ValueError, "unknown loss: %s" % loss_to_use)
        
        if self.architecture == "basic":
            # Basic RNN does not have a loop function in its API, so copying here.
            with vs.variable_scope("basic_rnn_seq2seq"):
                #tf.reset_default_graph()
                self.outputs_prev, self.enc_state = tf.contrib.rnn.static_rnn(self.cell, self.enc_in, dtype=tf.float32) # Encoder
                self.outputs, self.states = tf.contrib.legacy_seq2seq.rnn_decoder( self.dec_in, self.enc_state, self.cell, loop_function=lf ) # Decoder
        elif self.architecture == "tied":
            self.outputs, self.states = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq( self.enc_in, self.dec_in, self.cell, loop_function=lf )
        else:
            raise(ValueError, "Uknown architecture: %s" % architecture )
        #self.outputs = outputs
        '''
        with tf.variable_scope("rnn_layer", reuse=self.reuse):
            self.rnn_outputs, self.rnn_state = tf.nn.dynamic_rnn(self.cell,
                                                                 self.inputs_hidden,
                                                                 sequence_length=self.prediction_seq_len,
                                                                 initial_state=self.initial_states,
                                                                 dtype=tf.float32)
            self.prediction_representation = self.rnn_outputs
        '''
        #self.build_output_layer()
        if self.is_training or self.is_eval:
            self.build_loss()
        

    #def build_loss(self):
    #    with tf.name_scope("loss_angles"):
    #        loss_angles = tf.reduce_mean(tf.square(tf.subtract(self.dec_out, self.outputs)))+0*tf.reduce_mean(tf.square(tf.subtract(self.enc_prev[-1:-50:-1], self.outputs_prev[-1:-50:-1])))

    #    self.loss         = loss_angles
        #self.loss_summary = tf.summary.scalar('loss/loss', self.loss)
        #super(DummyModel, self).build_loss()
        
    def build_loss(self):
        with tf.name_scope("loss_angles"):
            
            target = tf.reshape(self.dec_out,[-1,self.input_size])
            out_put = tf.reshape(self.outputs,[-1,self.input_size])
            print("out_put_shape",tf.shape(out_put))
            target = tf.reshape(target,[-1,3])
            out_put = tf.reshape(out_put,[-1,3])
            
            angle1 = tf.sqrt(tf.reduce_sum(tf.square(out_put),1))
            angle1 = tf.reshape(angle1,[-1,1])
            angle = tf.concat([angle1,angle1],axis=1)
            angle = tf.concat([angle, angle1],axis=1)
            axis1 = out_put/angle
            
            angle2 = tf.sqrt(tf.reduce_sum(tf.square(target),1))
            angle2 = tf.reshape(angle2,[-1,1])
            angle_ = tf.concat([angle2,angle2],axis=1)
            angle_ = tf.concat([angle_, angle2],axis=1)
            axis2 = target/angle_
            
            def axis_angle_to_matrix(angle,axis):
                sin_axis = tf.sin(angle) * axis
                cos_angle = tf.cos(angle)
                cos1_axis = (1.0-cos_angle) * axis
                _, axis_y, axis_z = tf.unstack(axis, axis=-1)
                cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
                sin_axis_x, sin_axis_y,sin_axis_z = tf.unstack(sin_axis,axis=-1)
                tmp = cos1_axis_x * axis_y
                m01 = tmp - sin_axis_z
                m10 = tmp + sin_axis_z
                tmp = cos1_axis_x * axis_z
                m02 = tmp + sin_axis_y
                m20 = tmp - sin_axis_y
                tmp = cos1_axis_y * axis_z
                m12 = tmp - sin_axis_x
                m21 = tmp + sin_axis_x
                diag = cos1_axis * axis +cos_angle
                diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
                matrix = tf.stack((diag_x, m01, m02,
                                   m10, diag_y, m12,
                                   m20, m21, diag_z),axis=-1)
                output_shape = tf.concat((tf.shape(input=axis)[:-1],
                                          (3,3)),axis=-1)
                return tf.reshape(matrix,shape=output_shape)
            R1 = axis_angle_to_matrix(angle1,axis1)
            R2 = axis_angle_to_matrix(angle2,axis2)
            R1_t = tf.transpose(R1,[0,2,1])
            diff = (tf.trace(tf.matmul(R1_t,R2))-1.0)/2.0
            diff = tf.reshape(diff,[-1,1])
            cond1 = tf.cast((diff>=1), tf.float32)
            cond2 = tf.cast((diff<=-1),tf.float32)
            cond3 = tf.cast((tf.abs(diff)<1),tf.float32)
            diff_angle = tf.acos(tf.sign(diff)*(tf.abs(diff)-0.0001))
            self.loss = 180.0* tf.reduce_mean(tf.abs(diff_angle))/3.1415926

    def step(self, session):
        """
        Run a training or validation step of the model.
        Args:
          session: Tensorflow session object.
        Returns:
          A triplet of loss, summary update and predictions.
        """
        if self.is_training:
            # Training step.
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs,
                           self.parameter_update]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]
        else:
            # Evaluation step (no backprop).
            output_feed = [self.loss,
                           self.summary_update,
                           self.outputs]
            outputs = session.run(output_feed)
            return outputs[0], outputs[1], outputs[2]

    def sampled_step(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This still assumes
        that we have ground-truth available.
        Args:
          session: Tensorflow session object.
        Returns:
          Prediction with shape (batch_size, self.target_seq_len, feature_size), ground-truth targets, seed sequence and
          unique sample IDs.
        """
        assert self.is_eval, "Only works in evaluation mode."

        # Get the current batch.
        #batch = session.run(self.data_placeholders)
        #data_id = batch[C.BATCH_ID]
        #data_sample = batch[C.BATCH_INPUT]
        #targets = data_sample[:, self.source_seq_len:]

        #seed_sequence = data_sample[:, :self.source_seq_len]
        predictions = self.sample(session)

        return predictions[0], predictions[1], predictions[3], predictions[2]

    def predict(self, session):
        """
        Generates a sequence by feeding the prediction of time step t as input to time step t+1. This assumes no
        ground-truth data is available.
        Args:
            session: Tensorflow session object.

        Returns:
            Prediction with shape (batch_size, self.target_seq_len, feature_size), seed sequence and unique sample IDs.
        """
        # `sampled_step` is written such that it works when no ground-truth data is available, too.
        output_feed = [self.outputs,
                       self.prediction_inputs,
                       self.data_ids]
        outputs = session.run(output_feed)
        return outputs[0], outputs[1], outputs[2]

    def sample(self, session):
        """
        Generates `prediction_steps` may poses given a seed sequence.
        Args:
            session: Tensorflow session object.
            seed_sequence: A tensor of shape (batch_size, seq_len, feature_size)
            prediction_steps: How many frames to predict into the future.
            **kwargs:
        Returns:
            Prediction with shape (batch_size, prediction_steps, feature_size)
        """
        assert self.is_eval, "Only works in sampling mode."
        output_feed = [self.loss,
                       self.summary_update,
                       self.outputs,
                       self.decoder_outputs]
        outputs = session.run(output_feed)
        return outputs[0],outputs[1],outputs[2],outputs[3]
        '''
        one_step_seq_len = np.ones(seed_sequence.shape[0])

        # Feed the seed sequence to warm up the RNN.
        feed_dict = {self.prediction_inputs: seed_sequence,
                     self.prediction_seq_len: np.ones(seed_sequence.shape[0])*seed_sequence.shape[1]}
        state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)

        # Now create predictions step-by-step.
        prediction = prediction[:, -1:]
        predictions = [prediction]
        for step in range(prediction_steps-1):
            # get the prediction
            feed_dict = {self.prediction_inputs: prediction,
                         self.initial_states: state,
                         self.prediction_seq_len: one_step_seq_len}
            state, prediction = session.run([self.rnn_state, self.outputs], feed_dict=feed_dict)
            predictions.append(prediction)
        return np.concatenate(predictions, axis=1)
        '''
        
