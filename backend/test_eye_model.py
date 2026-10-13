import numpy as np
import tensorflow as tf

for path in ["models/eye_iron_model.keras", "models/eye_vita_model.keras"]:
    model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
    print("\n", path)
    print("  input shape :", model.input_shape)
    print("  output shape:", model.output_shape)
    dummy = np.random.randint(0, 255, (1, 224, 224, 3)).astype("float32")
    print("  sample out  :", model.predict(dummy, verbose=0))