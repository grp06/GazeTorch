import cv2
import numpy as np

def make_noise_field(
    h: int,
    w: int,
    rng: np.random.Generator,
    blotch_size: float = 9.0,
) -> np.ndarray:
    """
    Generates a low-frequency noise field with optional Gaussian smoothing.

    Parameters:
        h (int): Height of the image.
        w (int): Width of the image.
        rng (np.random.Generator): Random number generator.
        blotch_size (float): Gaussian sigma for smoothing.
        scale_blotches (int): Downsample factor for coarse blotches.

    Returns:
        np.ndarray: Normalized noise field of shape (h, w).
    """
    n = rng.standard_normal((h, w)).astype(np.float32)

    if blotch_size and blotch_size > 0:
        k = max(3, int(2 * round(blotch_size) + 1))
        n = cv2.GaussianBlur(n, (k, k), blotch_size)

    s = n.std()
    if s > 1e-6:
        n /= s

    # print(f"range: {np.amin(n)} to {np.amax(n)}")
    # cv2.imwrite("temp/noise_field.png", n * 55)

    return n


def add_ycrcb_noise(
    img_rgb: np.ndarray,
    chroma_strength: float = 15.0,
    chroma_blotch_size: float = 2.0,
    chroma_cr_cb_bias: tuple[float, float] = (1.0, 1.0),  # weight Cr vs Cb
    luma_sigma: float = 11.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Adds noise to YCrCb channels: chroma blotches to Cr/Cb and Gaussian grain to Y.

    Parameters:
        img_bgr (np.ndarray): Input BGR image.
        chroma_strength (float): Strength of chroma blotches.
        chroma_blotch_size (float): Size of chroma blotches (Gaussian sigma).
        chroma_scale_blotches (int): Downsample factor for coarse chroma blotches.
        chroma_cr_cb_bias (Tuple[float, float]): Weighting of Cr vs Cb noise.
        luma_sigma (float): Standard deviation of Gaussian noise for luminance.
        seed (Optional[int]): Random seed for reproducibility.

    Returns:
        np.ndarray: BGR image with combined YCrCb noise.
    """

    img_rgb = (img_rgb*255).astype(np.uint8)

    # assert img_bgr.dtype == np.uint8 and img_bgr.ndim == 3 and img_bgr.shape[2] == 3
    h, w = img_rgb.shape[:2]
    rng = np.random.default_rng(seed)

    # Convert to YCrCb and split channels
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Add Gaussian grain to luminance
    if luma_sigma > 0:
        luma_noise = rng.normal(loc=0.0, scale=luma_sigma, size=Y.shape).astype(np.float32)
        Y = np.clip(Y + luma_noise, 0, 255)

    # Add chroma blotches to Cr and Cb
    if chroma_strength > 0:
        n_cr = make_noise_field(h, w, rng, chroma_blotch_size) * chroma_strength * chroma_cr_cb_bias[0]
        n_cb = make_noise_field(h, w, rng, chroma_blotch_size) * chroma_strength * chroma_cr_cb_bias[1]
        Cr = np.clip(Cr + n_cr, 0, 255)
        Cb = np.clip(Cb + n_cb, 0, 255)

    # Merge and convert back to BGR
    out_ycrcb = cv2.merge([Y, Cr, Cb]).astype(np.uint8)

    out_noisy = cv2.cvtColor(out_ycrcb, cv2.COLOR_YCrCb2RGB)

    out_noisy = out_noisy.astype("float32")
    M = np.amax(out_noisy)
    if M>1e-3:
        out_noisy /= M
    
    return out_noisy


if __name__ == "__main__":
    img = cv2.imread(r"D:\py\ETH-XGaze\224\000.jpg", cv2.IMREAD_COLOR)

    import time
    ti = time.time()
    for _ in range(10):
        # out = add_chroma_blotches_bgr(
        #     img,
        #     strength=20,
        #     blotch_size=2,   # for bigger blobs 斑点大小
        #     # seed=42
        # )
        # img_chroma_grain = add_luma_gaussian_grain(out, sigma=11, seed=7)
        out = add_ycrcb_noise(img)

    to = time.time()
    print("Time:", to-ti)

    cv2.imwrite("temp/result_v3.png", out)
