import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
epsilon = 1e-7
cls_threshold = 0.8

def cls_cross_entropy(weights):
    #print(weights.shape)
    #weights = tf.convert_to_tensor(weights, np.float32)#K.variable(weights)
    def _cls_loss(y_true, y_pred):
        indicator = K.greater_equal(y_pred, cls_threshold)
        indicator = K.cast(indicator, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        smoother = K.pow((1 - y_pred), 0.5)
        result = -K.mean(weights * smoother * K.log(y_pred) * y_true)
        return result
    return _cls_loss


def deeplab_cls_cross_loss(weights):
    #print(weights.shape)
    #weights = tf.convert_to_tensor(weights, np.float32)#K.variable(weights)
    def _cls_loss(y_true, y_pred):
        indicator = K.greater_equal(y_pred, cls_threshold)
        indicator = K.cast(indicator, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        smoother = K.pow((1 - y_pred), 0.5)
        result = -K.mean(smoother * K.log(y_pred) * y_true)
        return result
    return _cls_loss


def detection_double_focal_loss_K(weight, fkg_focal_smoother, bkg_focal_smoother):
    """
    Binary crossentropy loss with focal.
    :param weight:
    :param focal_smoother:
    :return:
    """
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = y_pred[:, :, :, 1]
        y_true = y_true[:, :, :, 1]
        fkg_smooth = K.pow((1-y_pred), fkg_focal_smoother)
        bkg_smooth = K.pow(y_pred, bkg_focal_smoother)
        result = -K.mean(weight[1] * fkg_smooth * y_true * K.log(y_pred) +
                         weight[0] * bkg_smooth * (1-y_true) * K.log(1-y_pred))
        return result
    return _detection_loss



def classification_loss(weights, threshold=cls_threshold):

    def _classification_loss(y_true, y_pred):
        indicator = tf.greater_equal(y_pred, threshold, name='indicator_great')
        indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
        class_weights = tf.convert_to_tensor(weights, name='cls_weight_convert')
        class_weights = tf.cast(class_weights,tf.float32)
        # logits = tf.convert_to_tensor(y_pred, name='logits_convert', dtype=tf.float64)
        logits = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        logits = tf.cast(logits, tf.float32, name='logits_cast')
        """
        try:
            y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
        except ValueError:
            raise ValueError(
                 "indicator must have the same shape (%s vs %s)" %
                 (indicator.get_shape(), y_pred.get_shape()))
        """
        loss = -tf.reduce_mean(class_weights * indicator * tf.log(logits, name='logitslog'))
        return loss

    return _classification_loss


def joint_loss(det_weights, cls_joint_weights, joint_weights, cls_threshold = cls_threshold):
    def _joint_loss(y_true, y_pred):
        def _detection_loss(y_true, y_pred, det_weights):
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, det_weights))

        def _classification_loss(y_true, y_pred, cls_joint_weights, threshold):
            indicator = tf.greater_equal(y_pred, threshold, name='indicator_great')
            indicator = tf.cast(indicator, tf.float32, name='indicator_cast')
            class_weights = tf.convert_to_tensor(cls_joint_weights, name='cls_weight_convert')
            class_weights = tf.cast(class_weights, tf.float32)
            # logits = tf.convert_to_tensor(y_pred, name='logits_convert', dtype=tf.float64)
            logits = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
            logits = tf.cast(logits, tf.float32, name='logits_cast')
            """
            try:
                y_pred.get_shape().assert_is_compatible_with(indicator.get_shape())
            except ValueError:
                raise ValueError(
                     "indicator must have the same shape (%s vs %s)" %
                     (indicator.get_shape(), y_pred.get_shape())
                                )
                                """
            loss = -tf.reduce_mean(class_weights * indicator * tf.log(logits, name='logitslog'))
            return loss

        det_loss = _detection_loss(y_true, y_pred, det_weights)
        cls_loss = _classification_loss(y_true, y_pred, cls_joint_weights, cls_threshold)
        total_loss = tf.add(det_loss, tf.multiply(cls_loss, joint_weights))
        return total_loss
    return _joint_loss


def detection_focal_loss(weight, focal_smoother):
    def _detection_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, epsilon, 1-epsilon)
        weights = tf.convert_to_tensor(weight)
        weights = tf.cast(weights, tf.float32)
        focal_smooth = tf.cast(focal_smoother, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        smooth = tf.pow((1-y_pred), focal_smooth)
        result = -tf.reduce_mean(weights * (smooth * y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred)))
        return result
    return _detection_loss