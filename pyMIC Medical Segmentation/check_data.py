from pathlib import Path
import cv2
import matplotlib.pyplot as plt


image_path = Path(
    "data/images/12_HC.png"
)

label_path = Path(
    "data/labels/12_HC_seg.png"
)


# 读取
image = cv2.imread(
    str(image_path),
    0
)

mask = cv2.imread(
    str(label_path),
    0
)


# 显示

plt.figure(figsize=(12,4))


plt.subplot(1,3,1)
plt.title("Image")
plt.imshow(image,cmap="gray")
plt.axis("off")


plt.subplot(1,3,2)
plt.title("Mask")
plt.imshow(mask,cmap="gray")
plt.axis("off")


plt.subplot(1,3,3)
plt.title("Overlay")

plt.imshow(
    image,
    cmap="gray"
)

plt.imshow(
    mask,
    cmap="jet",
    alpha=0.4
)

plt.axis("off")


plt.show()