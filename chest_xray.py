# === Pneumonia Detection with Grad-CAM + LIME + SHAP ===
import os
import numpy as np
import seaborn as sns
import cv2
import matplotlib

# set backend BEFORE importing pyplot
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog

# === LIME & SHAP Imports ===
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap   # make sure `pip install shap` is done

# === Settings ===
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 3
MODEL_PATH = r"C:\Users\ASUS\Downloads\PUNEMONIA\pneumonia_model.h5"

# === Dataset Paths ===
train_dir = r"C:\Users\ASUS\Downloads\PUNEMONIA\Pneumonia detection\train"
val_dir   = r"C:\Users\ASUS\Downloads\PUNEMONIA\Pneumonia detection\val"
test_dir  = r"C:\Users\ASUS\Downloads\PUNEMONIA\Pneumonia detection\test"

# === Data Generators ===
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='binary',
    shuffle=False
)

# === Model Creation ===
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False,
                             input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# === Train or Load Model ===
if os.path.exists(MODEL_PATH):
    print(f"\n✅ Found saved model at {MODEL_PATH}, loading instead of training...\n")
    model = load_model(MODEL_PATH)
else:
    print("\n🚀 Training new model...\n")
    model = create_model()
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=EPOCHS)
    model.save(MODEL_PATH)
    print(f"\n💾 Model saved as {MODEL_PATH}\n")

# === Evaluate ===
loss, acc = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%\n")

y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype("int32")
print("\nClassification Report:\n", classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# === Grad-CAM Functions ===
def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(array, axis=0) / 255.0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return heatmap

def display_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)
    plt.figure(figsize=(8,8))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(superimposed_img); plt.title('Grad-CAM'); plt.axis('off')
    plt.show()

# === Prediction Wrapper for LIME ===
def predict_fn(imgs):
    imgs = np.array(imgs) / 255.0
    return model.predict(imgs)

# === GUI Upload with Grad-CAM + LIME + SHAP ===
def predict_uploaded_image():
    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Chest X-ray Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        print("❌ No file selected."); return

    # Prediction
    img_array = get_img_array(file_path, size=(IMG_SIZE, IMG_SIZE))
    prediction = model.predict(img_array)[0][0]
    print(f"🩺 Prediction: {'Pneumonia' if prediction > 0.5 else 'Normal'} "
          f"(Confidence: {prediction if prediction>0.5 else 1-prediction:.2f})")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1")
    display_gradcam(file_path, heatmap)

    # LIME
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(file_path, target_size=(IMG_SIZE, IMG_SIZE))
        ),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=500
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    plt.figure(figsize=(6,6))
    plt.imshow(mark_boundaries(temp/255.0, mask))
    plt.title("LIME Explanation"); plt.axis("off")
    plt.show()

    # SHAP
    print("\n🔍 Generating SHAP explanation (may take some time)...")
    background_data, _ = next(iter(val_generator))
    background_data = background_data[:20]   # small background for speed
    explainer = shap.GradientExplainer(model, background_data)
    shap_values = explainer.shap_values(img_array)
    shap.image_plot(shap_values, img_array, show=True)

# === Run ===
print("\n📂 Select an X-ray image to test...\n")
predict_uploaded_image()