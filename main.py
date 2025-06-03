import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils.data_loader import load_dataset, prepare_labels
from utils.model import build_model, iou_metric
from utils.visualization import display_prediction

# Load data
image_dir = './frames'
xml_dir = './annotations'
X, y_box, y_cls_raw = load_dataset(xml_dir, image_dir, augment_data=True)
X = X / 255.0

y_cls, class_names = prepare_labels(y_cls_raw)

X_train, X_val, y_box_train, y_box_val, y_cls_train, y_cls_val = train_test_split(
    X, y_box, y_cls, test_size=0.2, random_state=42
)

model = build_model(num_classes=len(class_names))
model.compile(
    optimizer='adam',
    loss={'class': 'categorical_crossentropy', 'bbox': 'mse'},
    loss_weights={'class': 1.0, 'bbox': 15.0},
    metrics={'class': 'accuracy', 'bbox': [iou_metric, 'mse']}
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

model.fit(
    X_train,
    {'class': y_cls_train, 'bbox': y_box_train},
    validation_data=(X_val, {'class': y_cls_val, 'bbox': y_box_val}),
    epochs=25,
    batch_size=11,
    callbacks=callbacks
)

preds = model.predict(X_val[:5])
for i in range(5):
    display_prediction(X_val[i], y_box_val[i], preds[1][i], np.argmax(y_cls_val[i]), np.argmax(preds[0][i]), class_names)
