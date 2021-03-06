import tensorflow as tf


class sigmoid_loss:
    def __init__(self, skew = 100):
        super().__init__()
        self.skew = 100
        self.mask = tf.keras.layers.Masking(mask_value=-1)

    def __call__(self, y_pred, y_true):
        attention_mask = self.mask.compute_mask(y_true)
        attention_mask = tf.expand_dims(tf.cast(attention_mask,tf.float32),-1)
        y_true = tf.where(y_true==-1,tf.zeros_like(y_true),y_true)
        y_true = tf.cast(y_true, tf.float32)
        mask = ((1.0-y_true) + y_true *(self.skew))*attention_mask
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)*mask)
        return loss, tf.nn.sigmoid(y_pred)

    def get_metrics(self):
        METRICS = {'train': [
                    tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalseNegatives(name='fn'), 
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                    ], 
                    'val': [
                    tf.keras.metrics.TruePositives(name='tp'),
                    tf.keras.metrics.FalsePositives(name='fp'),
                    tf.keras.metrics.TrueNegatives(name='tn'),
                    tf.keras.metrics.FalseNegatives(name='fn'), 
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc'),
                ]  }
        return METRICS

    def get_output(self, outputs):
        return tf.nn.sigmoid(outputs)


class cross_entropy:
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.mask = tf.keras.layers.Masking(mask_value=-1)
    
    def __call__(self, y_pred, y_true):
        attention_mask = self.mask.compute_mask(tf.expand_dims(y_true,-1))
        y_true = tf.where(y_true==-1,tf.zeros_like(y_true),y_true)
        loss = self.loss_fn(y_true, y_pred,attention_mask)
        return loss, tf.nn.softmax(y_pred)

    def get_metrics(self):
        METRICS = {
            "train":
                [tf.keras.metrics.SparseCategoricalAccuracy(
                name='sparse_categorical_accuracy_train')],
            "val":
                [tf.keras.metrics.SparseCategoricalAccuracy(
                name='sparse_categorical_accuracy_val')]
            }
        return METRICS
    
    def get_output(self, outputs):
        return tf.nn.softmax(outputs)


class cross_entropy_weighted:
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    def __call__(self, y_pred, y_true):
        labels = tf.cast(tf.squeeze(y_true[:,:,:-1]), tf.int32)
        sample_weights = y_true[:,:,-1]
        loss = self.loss_fn(labels, y_pred, sample_weights)
        return loss, tf.nn.softmax(y_pred)

    def get_metrics(self):
        METRICS = {
            "train":
                [tf.keras.metrics.SparseCategoricalAccuracy(
                name='sparse_categorical_accuracy_train')],
            "val":
                [tf.keras.metrics.SparseCategoricalAccuracy(
                name='sparse_categorical_accuracy_val')]
            }
        return METRICS
    
    def get_output(self, outputs):
        return tf.nn.softmax(outputs)

class kl:
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.KLDivergence()
        self.mask = tf.keras.layers.Masking(mask_value=-1.)
    
    def __call__(self, y_pred, y_true):
        attention_mask = self.mask.compute_mask(y_true)
        attention_mask = tf.expand_dims(tf.cast(attention_mask,tf.float32),-1)
        y_true = tf.where(y_true==-1.,tf.zeros_like(y_true),y_true)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.nn.softmax(y_pred)
        loss = self.loss_fn(y_true, y_pred,attention_mask)
        return loss, y_pred

    def get_metrics(self):
        METRICS = {
            "train":
                [tf.keras.metrics.KLDivergence(
                name='kl_divergence_train')],
            "val":
                [tf.keras.metrics.KLDivergence(
                name='kl_divergence_val')]
            }
        return METRICS

    def get_output(self, outputs):
        return tf.nn.softmax(outputs)        


class kl_weighted:
    def __init__(self):
        super().__init__()
        self.loss_fn = tf.keras.losses.KLDivergence()
    
    def __call__(self, y_pred, y_true):
        labels = y_true[:,:,:-1]
        sample_weights = y_true[:,:,-1]
        y_pred = tf.nn.softmax(y_pred)
        loss = self.loss_fn(labels, y_pred, sample_weights)
        return loss, y_pred

    def get_metrics(self):
        METRICS = {
            "train":
                [tf.keras.metrics.KLDivergence(
                name='kl_divergence_train')],
            "val":
                [tf.keras.metrics.KLDivergence(
                name='kl_divergence_val')]
            }
        return METRICS

    def get_output(self, outputs):
        return tf.nn.softmax(outputs)        
