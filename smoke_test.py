import json, joblib, numpy as np, pandas as pd, tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class AddPosition(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(1, input_shape[1], input_shape[2]), initializer="random_normal")
    def call(self, x):
        return x + self.pos

C = "checkpoints"
cfg = json.load(open(f"{C}/finalist_config.json"))
model = tf.keras.models.load_model(f"{C}/finalist.keras")
pre = joblib.load(f"{C}/preprocessing.joblib")
idx = np.load(f"{C}/split_indices.npz")["test"][:8]
data = pd.read_csv("FordA_TRAIN.tsv", sep="\t", header=None).to_numpy(np.float32)
X, y = data[idx, 1:], (data[idx, 0] == 1).astype(int)
X = pre["scaler"].transform(pre["imputer"].transform(X)).astype("float32")
W, stride = cfg["window"], cfg["stride"]
starts = list(range(0, X.shape[1] - W + 1, stride))
Xw = np.stack([X[:, s:s + W] for s in starts], axis=1).reshape(-1, W, 1)
if cfg["input_kind"] == "flat":
    Xw = Xw[..., 0]
prob = model.predict(Xw, verbose=0).ravel().reshape(len(X), -1).mean(axis=1)
assert prob.shape == (8,) and ((prob >= 0) & (prob <= 1)).all()
print("SMOKE TEST PASSED | finalist:", cfg["family"], "| accuracy on 8 test series:", float(((prob >= 0.5) == y).mean()))
