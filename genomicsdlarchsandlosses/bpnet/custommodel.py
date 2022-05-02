import tensorflow as tf
import tensorflow.keras.backend as kb

from genomicsdlarchsandlosses.bpnet.losses import multinomial_nll

from tensorflow import keras
from tensorflow.keras import Model


class CustomModel(Model):

    def __init__(self, num_output_tracks, loss_weights, **kwargs):

        # call the base class with inputs and outputs
        super(CustomModel, self).__init__(**kwargs)
        
        # number of output tracks
        self.num_output_tracks = num_output_tracks
        
        # weights for the profile mnll and logcounts mse losses
        self.loss_weights = loss_weights
        
        # object to track overall mean loss per epoch
        self.loss_tracker = keras.metrics.Mean(name="loss")


    def _get_loss(self, x, y, sample_weights, training=True):
        # boolean mask for sample weights != 0
        boolean_mask = tf.math.greater_equal(sample_weights, 1.0)
                
        y_pred = self(x, training=training)  # Forward pass

        # logcounts mse loss without sample weights
        mse_loss = keras.losses.mean_squared_error(
            y['logcounts_predictions'], y_pred[1])

        # for mnll loss we mask out samples with weight == 0.0
        _y = tf.boolean_mask(y['profile_predictions'], boolean_mask)
        _y_pred = tf.boolean_mask(y_pred[0], boolean_mask)

        def _zero_constant():
            return kb.constant(0)
    
        def _multinomial_nll():
            total_mnll_loss = 0
            for i in range(self.num_output_tracks):
                loss = multinomial_nll(_y[..., i], _y_pred[..., i])
                total_mnll_loss += loss
            return total_mnll_loss
            
        total_mnll_loss = tf.cond(tf.equal(tf.size(_y), 0), 
                   _zero_constant,
                   lambda:  _multinomial_nll())

        # sum of weighted losses
        loss =  (self.loss_weights[0] * total_mnll_loss) + \
            (self.loss_weights[1] * mse_loss)

        return loss, total_mnll_loss, mse_loss
            
    def train_step(self, data):
        x, y, sample_weights = data
    
        with tf.GradientTape() as tape:
            loss, total_mnll_loss, mse_loss = \
                self._get_loss(x, y, sample_weights)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(),
                "batch_loss": loss,
                "profile_predictions_loss": total_mnll_loss, 
                "logcounts_predictions_loss": mse_loss}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]
    
    
    def test_step(self, data):
        # Unpack the data
        x, y, sample_weights = data
        
        loss, total_mnll_loss, mse_loss = \
            self._get_loss(x, y, sample_weights, training=False)
            
        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result(),
                "batch_loss": loss,
                "profile_predictions_loss": total_mnll_loss, 
                "logcounts_predictions_loss": mse_loss}
