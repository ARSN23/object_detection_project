import tensorflow as tf

def iou_metric(y_true, y_pred):
    xA = tf.maximum(y_true[:,0], y_pred[:,0])
    yA = tf.maximum(y_true[:,1], y_pred[:,1])
    xB = tf.minimum(y_true[:,2], y_pred[:,2])
    yB = tf.minimum(y_true[:,3], y_pred[:,3])

    interArea = tf.maximum((xB - xA), 0) * tf.maximum((yB - yA), 0)
    boxAArea = (y_true[:,2] - y_true[:,0]) * (y_true[:,3] - y_true[:,1])
    boxBArea = (y_pred[:,2] - y_pred[:,0]) * (y_pred[:,3] - y_pred[:,1])

    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return tf.reduce_mean(iou)


def build_model(num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    class_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='class')(x)
    bbox_output = tf.keras.layers.Dense(4, activation='sigmoid', name='bbox')(x)

    return tf.keras.Model(inputs, outputs=[class_output, bbox_output])
