import cv2
import numpy as np
import matplotlib.pyplot as plt

AREA_PRATO_CM2 = 480


def gerar_heightmap(
    mask,
    plate_area,
    curve="power",
    gamma=0.5,
    alpha=4.0,
    smooth_px=5
):
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_RGB2GRAY)

    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    else:
        mask = mask.astype(np.float32)

        if mask.max() <= 1.0:
            mask = mask * 255

        mask = mask.astype(np.uint8)

    _, mask_bin = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY
    )

    if not np.any(mask_bin > 0):
        heightmap = np.zeros_like(mask_bin, dtype=np.uint8)

        metrics = {
            "area_px2": 0,
            "total_height": 0.0,
            "mean_height": 0.0,
            "max_height": 0.0,
            "volume_estimado": 0.0,
        }

        return heightmap, metrics

    dist = cv2.distanceTransform(
        mask_bin,
        cv2.DIST_L2,
        maskSize=5
    )

    d_max = dist[mask_bin > 0].max()
    dist_norm = dist / d_max

    if curve == "power":
        morro = dist_norm ** gamma

    elif curve == "exp":
        morro = (
            np.exp(alpha * dist_norm) - 1
        ) / (
            np.exp(alpha) - 1
        )

    else:
        raise ValueError("curve deve ser 'power' ou 'exp'.")

    morro[mask_bin == 0] = 0

    heightmap = (morro * 255).astype(np.uint8)

    if smooth_px is not None and smooth_px > 1:
        k = smooth_px if smooth_px % 2 == 1 else smooth_px + 1

        heightmap = cv2.GaussianBlur(
            heightmap,
            (k, k),
            0
        )

        heightmap[mask_bin == 0] = 0

    pixels = heightmap[mask_bin > 0].astype(float)

    area_px = int(len(pixels))
    total_height = float(pixels.sum())
    mean_height = float(pixels.mean())
    max_height = float(pixels.max())

    area_cm = area_px * (AREA_PRATO_CM2 / plate_area)

    volume_estimado = area_cm * (mean_height * 3 / 255)

    metrics = {
        "area_px2": area_px,
        "total_height": total_height,
        "mean_height": mean_height,
        "max_height": max_height,
        "volume_estimado": volume_estimado,
    }

    return heightmap, metrics


def visualizar_heightmap(
    mask,
    heightmap,
    metrics=None,
    nome="alimento"
):
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_RGB2GRAY)

    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255
    else:
        mask = mask.astype(np.uint8)

    _, mask_bin = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY
    )

    step = max(1, min(heightmap.shape[:2]) // 200)

    hm_small = heightmap[::step, ::step].astype(float)
    mask_small = mask_bin[::step, ::step]

    rows, cols = hm_small.shape
    X, Y = np.meshgrid(
        np.arange(cols),
        np.arange(rows)
    )

    Z = hm_small.copy()
    Z[mask_small == 0] = 0

    fig = plt.figure(figsize=(16, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(mask_bin, cmap="gray")
    ax1.set_title(f"Máscara - {nome}")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(heightmap, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("Heightmap")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    im = ax3.imshow(heightmap, cmap="jet", vmin=0, vmax=255)
    ax3.set_title("Heightmap em pseudo-cor")
    ax3.axis("off")

    plt.colorbar(
        im,
        ax=ax3,
        fraction=0.046,
        pad=0.04
    )

    plot_res = 1

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    ax4.plot_surface(
        X,
        Y,
        Z,
        cmap="jet",
        rstride=plot_res,
        cstride=plot_res,
        linewidth=0,
        antialiased=False
    )

    ax4.set_title("Superfície 3D")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Altura")
    ax4.view_init(elev=35, azim=-60)

    plt.suptitle(f"Heightmap - {nome}", fontsize=14)
    plt.tight_layout()
    plt.show()