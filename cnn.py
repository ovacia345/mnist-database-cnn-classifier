import numpy as np
import tensorflow as tf


def cnn_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[7, 7],
        strides=1,
        padding="same",
        activation=tf.nn.relu
    )
    maxpool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    conv2 = tf.layers.conv2d(
        inputs=maxpool1,
        filters=72,
        kernel_size=[3, 3],
        strides=1,
        padding="same",
        activation=tf.nn.relu
    )
    maxpool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    maxpool2_flat = tf.reshape(maxpool2, [-1, 7 * 7 * 72])
    dropout = tf.layers.dropout(
        inputs=maxpool2_flat,
        rate=0.5
    )
    logits = tf.layers.dense(
        inputs=dropout,
        units=10
    )

    predictions = {
        'class': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits=logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                           logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['class']
        )
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        eval_metric_ops=eval_metric_ops
    )


NUM_TRAINING_STEPS = 20000
N_ITER = 10


def main(args):
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_fn,
        model_dir=".\\cnn_data\\"
    )

    mnist_train = np.load('data\\training_set.npz')
    train_data = mnist_train['images'].astype(np.float32)
    train_labels = mnist_train['labels'].astype(np.int32)

    mnist_test = np.load('data\\test_set.npz')
    eval_data = mnist_test['images'].astype(np.float32)
    eval_labels = mnist_test['labels'].astype(np.int32)

    accuracy_list = np.zeros([NUM_TRAINING_STEPS // N_ITER], np.float32)
    loss_list = np.zeros([NUM_TRAINING_STEPS // N_ITER], np.float32)
    for i in range(NUM_TRAINING_STEPS // N_ITER):
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_data},
            y=train_labels,
            batch_size=128,
            num_epochs=None,
            shuffle=True
        )
        mnist_classifier.train(
            input_fn=train_input_fn,
            steps=N_ITER
        )

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

        accuracy_list[i] = eval_results['accuracy']
        loss_list[i] = eval_results['loss']

        print("step:{0}\tacc: {1}\tloss: {2}".format(
            eval_results['global_step'],
            eval_results['accuracy'],
            eval_results['loss']
        ))

    np.savez_compressed('training_data.npz', accuracy=accuracy_list,
                        loss=loss_list)


if __name__ == '__main__':
    tf.app.run()
