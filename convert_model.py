import tensorflow as tf

print("Loading model...")

# load without optimizer
model = tf.keras.models.load_model(
    "model/tomato_disease_model.h5",
    compile=False
)

print("Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# optimization for smaller model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model/plant_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model created successfully!")
