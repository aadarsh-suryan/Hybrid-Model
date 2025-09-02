# hybrid_with_gradcam.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Concatenate, BatchNormalization, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ----------------------------
# Configurable defaults
# ----------------------------
DEFAULT_DATASET = r"C:\Users\abhis\OneDrive\c++\Hybrid Model\datasets"
DEFAULT_OUTDIR = r"C:\Users\abhis\OneDrive\c++\Hybrid Model\output"
IMG_SIZE = 224

# ----------------------------
# Build hybrid model
# ----------------------------
def build_hybrid_vgg_resnet(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=3, freeze_backbones=True):
    # Shared input tensors
    vgg_input = Input(shape=input_shape, name="vgg_input")
    resnet_input = Input(shape=input_shape, name="resnet_input")

    # VGG19 branch
    vgg_base = VGG19(weights="imagenet", include_top=False, input_tensor=vgg_input)
    # ResNet50 branch
    resnet_base = ResNet50(weights="imagenet", include_top=False, input_tensor=resnet_input)

    if freeze_backbones:
        for layer in vgg_base.layers:
            layer.trainable = False
        for layer in resnet_base.layers:
            layer.trainable = False

    vgg_feat = GlobalAveragePooling2D(name="vgg_gap")(vgg_base.output)
    res_feat = GlobalAveragePooling2D(name="resnet_gap")(resnet_base.output)

    merged = Concatenate(name="concat_features")([vgg_feat, res_feat])
    merged = BatchNormalization(name="bn_merge")(merged)
    merged = Dense(512, activation="relu", name="fc1")(merged)
    merged = Dropout(0.5, name="drop1")(merged)
    merged = Dense(256, activation="relu", name="fc2")(merged)
    merged = Dropout(0.3, name="drop2")(merged)
    out = Dense(num_classes, activation="softmax", name="predictions")(merged)

    model = Model(inputs=[vgg_input, resnet_input], outputs=out, name="vgg_resnet_hybrid")
    return model

# ----------------------------
# Grad-CAM utility
# ----------------------------
def find_last_conv_layer(model):
    # Heuristic: find the last layer with 4D output shape and 'conv' in name
    for layer in reversed(model.layers):
        out_shape = getattr(layer, "output_shape", None)
        if out_shape and len(out_shape) == 4 and "conv" in layer.name:
            return layer.name
    # Fallback: last 4D layer
    for layer in reversed(model.layers):
        out_shape = getattr(layer, "output_shape", None)
        if out_shape and len(out_shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found in the model.")

def make_gradcam_heatmap(model, img_inputs, last_conv_layer_name, pred_index=None):
    """
    model: compiled or uncompiled keras Model
    img_inputs: list or tuple of numpy arrays shaped for model inputs, each (1,H,W,3)
                for our hybrid model supply [x,x] where x is the preprocessed image
    last_conv_layer_name: string
    pred_index: int or None
    returns: heatmap np.array (H_conv, W_conv) normalized 0..1
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_inputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[..., tf.newaxis]
    conv_outputs = tf.multiply(conv_outputs, pooled_grads)
    heatmap = tf.reduce_sum(conv_outputs, axis=-1)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0)
    maxval = tf.reduce_max(heatmap)
    if maxval == 0:
        return np.zeros((heatmap.shape[0], heatmap.shape[1]))
    heatmap /= maxval
    return heatmap.numpy()

def save_gradcam_overlay(orig_img_path, heatmap, output_path, alpha=0.4):
    """
    orig_img_path: path to original image file
    heatmap: 2D array normalized 0..1
    output_path: where to save overlay image (jpg/png)
    """
    orig = cv2.imread(orig_img_path)
    if orig is None:
        raise ValueError("Could not read image for overlay: " + orig_img_path)
    heatmap_resized = cv2.resize((heatmap * 255).astype("uint8"), (orig.shape[1], orig.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig, 1 - alpha, heatmap_color, alpha, 0)
    cv2.imwrite(output_path, overlay)
    return output_path

# ----------------------------
# Preprocess helpers
# ----------------------------
def load_and_preprocess_image(path, target_size=(IMG_SIZE, IMG_SIZE)):
    pil = Image.open(path).convert("RGB")
    pil = pil.resize(target_size)
    arr = np.array(pil).astype("float32") / 255.0
    return arr

# ----------------------------
# Training pipeline
# ----------------------------

def train_pipeline(dataset_dir, outdir, batch_size=32, epochs=20, fine_tune_layers=40):
    # =============================
    # 1. DATA GENERATORS
    # =============================
    img_height, img_width = 224, 224

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=8,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = train_generator.num_classes

    # =============================
    # 2. HYBRID MODEL (VGG19 + ResNet)
    # =============================

    input_tensor = Input(shape=(img_height, img_width, 3))

    vgg = VGG19(weights='imagenet', include_top=False, input_tensor=input_tensor)
    resnet = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

    # Freeze all layers first
    for layer in vgg.layers:
        layer.trainable = False
    for layer in resnet.layers:
        layer.trainable = False

    vgg_out = GlobalAveragePooling2D()(vgg.output)
    resnet_out = GlobalAveragePooling2D()(resnet.output)

    merged = Concatenate()([vgg_out, resnet_out])
    x = Dense(512, activation='relu')(merged)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)

    # =============================
    # 3. COMPILE & INITIAL TRAINING
    # =============================
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath="best_model.h5", monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]

    print("Starting initial training (frozen backbones)...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks
    )

    # =============================
    # 4. FINE-TUNING
    # =============================
    fine_tune_layers = 30 # Number of layers to fine-tune from the end
    print(f"Starting fine-tuning last {fine_tune_layers} layers...")
    for layer in vgg.layers[-fine_tune_layers:]:
        layer.trainable = True
    for layer in resnet.layers[-fine_tune_layers:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    fine_tune_history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=callbacks
    )

    return model, train_generator.class_indices


    # Save model and history
    final_model_path = os.path.join(out_dir, "hybrid_vgg_resnet_finetuned.h5")
    model.save(final_model_path)
    np.save(os.path.join(out_dir, "history_initial.npy"), history.history)
    np.save(os.path.join(out_dir, "history_finetune.npy"), ft_history.history)
    print("Training complete. Model saved to:", final_model_path)
    return final_model_path, train_gen.class_indices

# ----------------------------
# Inference + Grad-CAM
# ----------------------------
def infer_and_gradcam(model_paths, image_path, out_dir, tta=0, class_names=None):
    os.makedirs(out_dir, exist_ok=True)
    # load models (ensemble support)
    models = []
    for p in model_paths:
        if not os.path.exists(p):
            raise FileNotFoundError("Model not found: " + p)
        models.append(load_model(p, compile=False))

    # preprocess image
    arr = load_and_preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    # prepare inputs for hybrid model: both branches receive same image
    inp = [np.expand_dims(arr, axis=0), np.expand_dims(arr, axis=0)]

    # predict (simple average across models; TTA option per-model)
    probs_list = []
    for m in models:
        if tta and tta > 0:
            # TTA: generate small augmented variants
            tta_preds = []
            for i in range(tta):
                aug = arr.copy()
                if np.random.rand() < 0.5:
                    aug = np.fliplr(aug)
                angle = np.random.uniform(-8, 8)
                M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), angle, 1.0)
                aug = cv2.warpAffine((aug*255).astype('uint8'), M, (IMG_SIZE, IMG_SIZE)).astype('float32')/255.0
                pred = m.predict([np.expand_dims(aug,0), np.expand_dims(aug,0)])[0]
                tta_preds.append(pred)
            probs_list.append(np.mean(tta_preds, axis=0))
        else:
            probs_list.append(models[0].predict(inp)[0] if len(models)==1 else m.predict(inp)[0])

    avg_prob = np.mean(np.array(probs_list), axis=0)
    predicted_idx = int(np.argmax(avg_prob))
    confidence = float(avg_prob[predicted_idx])

    if class_names is None:
        # try to infer names from training folders (not guaranteed)
        class_names = [f"class_{i}" for i in range(len(avg_prob))]

    predicted_label = class_names[predicted_idx]
    print(f"Prediction: {predicted_label}  Confidence: {confidence*100:.2f}%")

    # Grad-CAM using first model
    model_for_gradcam = models[0]
    # find last conv layer
    last_conv = find_last_conv_layer(model_for_gradcam)
    print("Using last conv layer for Grad-CAM:", last_conv)

    heatmap = make_gradcam_heatmap(model_for_gradcam, inp, last_conv, pred_index=predicted_idx)
    out_gradcam = os.path.join(out_dir, "gradcam_overlay.jpg")
    save_gradcam_overlay(image_path, heatmap, out_gradcam, alpha=0.45)
    print("Grad-CAM overlay saved to:", out_gradcam)

    return predicted_label, confidence, avg_prob

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train","infer"], default="train")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--model_path", default=None, help="If infer mode single model path (optional)")
    p.add_argument("--model_paths", nargs="+", default=None, help="If infer mode multiple models for ensemble")
    p.add_argument("--image", default=None, help="Image path for inference")
    p.add_argument("--tta", type=int, default=0, help="Number of TTA augmentations for inference (optional)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        model_path, class_indices = train_pipeline(args.dataset, args.outdir, batch_size=args.batch_size, epochs=args.epochs, fine_tune_layers=40)
        print("Trained model:", model_path)
        print("Class indices:", class_indices)
    else:
        if not args.image:
            print("Provide --image for inference mode")
            return
        # choose model(s)
        models = []
        if args.model_paths:
            models = args.model_paths
        elif args.model_path:
            models = [args.model_path]
        else:
            # try defaults in outdir
            cand = [os.path.join(args.outdir, "hybrid_vgg_resnet_finetuned.h5"),
                    os.path.join(args.outdir, "hybrid_vgg_resnet_finetuned.h5"),
                    os.path.join(args.outdir, "best_hybrid.h5")]
            models = [c for c in cand if os.path.exists(c)]
            if not models:
                print("No model provided and no default model found in outdir.")
                return

        # attempt to infer class names
        class_names = None
        try:
            class_names = sorted(next(os.walk(args.dataset))[1])
        except Exception:
            class_names = None

        pred_label, conf, probs = infer_and_gradcam(models, args.image, args.outdir, tta=args.tta, class_names=class_names)
        print("Done.")

if __name__ == "__main__":
    main()


# Accuracy plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Loss plot
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid(True)
plt.show()