import matplotlib.pyplot as plt
import cv2

def display_prediction(image, true_box, pred_box, true_class, pred_class, class_names):
    h, w = image.shape[:2]
    image = cv2.cvtColor((image * 255).astype('uint8'), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)

    x1, y1, x2, y2 = [int(coord * w if i % 2 == 0 else coord * h) for i, coord in enumerate(true_box)]
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='green', facecolor='none', linewidth=2))
    ax.text(x1, y1 - 10, f"GT: {class_names[true_class]}", color='green', fontsize=10)

    px1, py1, px2, py2 = [int(coord * w if i % 2 == 0 else coord * h) for i, coord in enumerate(pred_box)]
    ax.add_patch(plt.Rectangle((px1, py1), px2 - px1, py2 - py1, edgecolor='red', facecolor='none', linewidth=2))
    ax.text(px1, py1 - 30, f"Pred: {class_names[pred_class]}", color='red', fontsize=10)

    plt.axis('off')
    plt.show()
