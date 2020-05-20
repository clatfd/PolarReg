from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context
import os
import tensorflow as tf
import keras

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.training_log_dir = os.path.join(log_dir, 'training')
        self.val_dscs = []
        super(TrainValTensorBoard, self).__init__(self.training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            #print(val_logs)
            for name, value in val_logs.items():
                if name == 'batch_loss':
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value#.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)

                if name == 'epoch_loss':
                    summary = tf.Summary()
                    summary_value = summary.value.add()
                    if len(self.val_dscs):
                        summary_value.simple_value = self.val_dscs[-1]
                    else:
                        summary_value.simple_value = 0
                    summary_value.tag = 'DSC'
                    self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

