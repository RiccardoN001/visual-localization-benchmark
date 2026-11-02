# Runtime patches for the upstream OmniGlue code (github.com/google-research/omniglue).
# They change speed only: matches are identical to the original implementation.
import numpy as np
import tensorflow as tf
import torch


def get_dino_descriptors_fast(dino_features, keypoints, height, width, feature_dim):
    """Vectorized version of omniglue.dino_extract.get_dino_descriptors.

    Upstream converts the whole feature map to numpy once per keypoint. This version does the same
    bilinear lookup in a single vectorized operation and reproduces the upstream behaviour exactly:
    the second sample along an axis exists only if the coordinate is not an integer and not on the
    last row/column, and the weights are not renormalized when it is missing.
    """
    from omniglue import dino_extract

    height_1d = tf.reshape(height, [1])
    width_1d = tf.reshape(width, [1])
    height_resized, width_resized = dino_extract._preprocess_shape(
        height_1d, width_1d, image_size_max=630, h_down_rate=14, w_down_rate=14)
    height_feat = height_resized // 14
    width_feat = width_resized // 14

    size_feature = tf.concat([height_feat, width_feat, tf.reshape(feature_dim, [1])], axis=0)
    dmap = np.asarray(tf.reshape(dino_features, size_feature))   # (Hf, Wf, D)

    img_size = tf.cast(tf.concat([width_1d, height_1d], axis=0), tf.float32)
    feature_size = tf.cast(tf.concat([width_feat, height_feat], axis=0), tf.float32)
    kp = np.asarray(keypoints / tf.expand_dims(img_size, axis=0)
                    * tf.expand_dims(feature_size, axis=0), dtype=np.float64)
    Hf, Wf = dmap.shape[0], dmap.shape[1]

    x, y = kp[:, 0], kp[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    has_x1 = (x != x0) & (x < Wf - 1)
    has_y1 = (y != y0) & (y < Hf - 1)
    x1 = np.where(has_x1, x0 + 1, x0)
    y1 = np.where(has_y1, y0 + 1, y0)

    x0c, x1c = np.clip(x0, 0, Wf - 1), np.clip(x1, 0, Wf - 1)
    y0c, y1c = np.clip(y0, 0, Hf - 1), np.clip(y1, 0, Hf - 1)
    wx0 = 1.0 - np.abs(x - x0)
    wy0 = 1.0 - np.abs(y - y0)
    wx1 = np.where(has_x1, 1.0 - np.abs(x - x1), 0.0)
    wy1 = np.where(has_y1, 1.0 - np.abs(y - y1), 0.0)

    out = ((wx0 * wy0)[:, None] * dmap[y0c, x0c]
           + (wx1 * wy0)[:, None] * dmap[y0c, x1c]
           + (wx0 * wy1)[:, None] * dmap[y1c, x0c]
           + (wx1 * wy1)[:, None] * dmap[y1c, x1c])
    return tf.convert_to_tensor(out, dtype=tf.float32)


def _dino_forward_on_device(self, image):
    # Same steps as DINOExtract.forward, with the input moved to the model device.
    image = self._resize_input_image(image)
    x = self._process_image(image).unsqueeze(0).float().to(self.device)
    features = self.extract_feature(x)
    return features.squeeze(0).permute(1, 2, 0).cpu().numpy()


def patch_omniglue(og, device):
    """Apply both patches: vectorized descriptor lookup, and DINOv2 on the GPU instead of the CPU."""
    from omniglue import dino_extract

    dino_extract.get_dino_descriptors = get_dino_descriptors_fast
    dino = og.dino_extract
    dino.device = torch.device(device)
    dino.model = dino.model.to(dino.device)
    dino.forward = _dino_forward_on_device.__get__(dino)
