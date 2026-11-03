"""Gradio demo: brain tumour classification on an MRI slice (research prototype, not for clinical use)."""
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr

IMG = 256
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # alphabetical, as in training
LABELS = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary tumour"}
MODEL = tf.keras.models.load_model("brain_tumor_effnetv2s.keras")
DISCLAIMER = ("Research prototype trained on a public MRI dataset. "
              "Not a medical device. Not for clinical use.")


def grad_cam(x):
    eff = MODEL.get_layer("efficientnetv2-s")
    gm = tf.keras.models.Model(eff.inputs, [eff.get_layer("top_activation").output, eff.output])
    with tf.GradientTape() as tape:
        conv, feat = gm(x)
        probs = MODEL.layers[-1](tf.reduce_mean(feat, axis=(1, 2)))
        k = tf.argmax(probs[0])
        loss = probs[:, k]
    g = tape.gradient(loss, conv)
    cam = tf.maximum(tf.reduce_sum(conv[0] * tf.reduce_mean(g, axis=(0, 1, 2)), -1), 0)
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy()


def predict(image):
    if image is None:
        return {}, None
    rgb = np.array(image.convert("RGB").resize((IMG, IMG)), dtype=np.float32)
    x = tf.constant(rgb[None])
    p = MODEL.predict(x, verbose=0)[0]
    overlay = None
    try:
        cam = cv2.resize(grad_cam(x), (IMG, IMG))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
        overlay = np.uint8(0.6 * rgb + 0.4 * heat)
    except Exception:
        pass
    return {LABELS[c]: float(v) for c, v in zip(CLASSES, p)}, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Brain MRI slice"),
    outputs=[gr.Label(label="Model output"), gr.Image(label="Grad-CAM (where the model looked)")],
    title="Brain tumour MRI classification (EfficientNetV2-S)",
    description=DISCLAIMER,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
