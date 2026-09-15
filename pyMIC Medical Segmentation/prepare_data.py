from pathlib import Path
import cv2
import shutil


# ==============================
# 路径
# ==============================

source_dir = Path(
    "data/HC18/training_set"
)

image_dir = Path(
    "data/images"
)

label_dir = Path(
    "data/labels"
)


image_dir.mkdir(exist_ok=True)
label_dir.mkdir(exist_ok=True)


# ==============================
# 遍历所有文件
# ==============================

count = 0


for file in source_dir.iterdir():

    name = file.name


    # --------------------------
    # 处理原始超声图
    # --------------------------

    if (
        name.endswith(".png")
        and "_Annotation" not in name
    ):

        shutil.copy(
            file,
            image_dir / name
        )


    # --------------------------
    # 处理标注
    # --------------------------

    elif "_Annotation" in name:


        # 读取轮廓图
        annotation = cv2.imread(
            str(file),
            0
        )


        # 创建空mask
        mask = annotation.copy()


        # 找轮廓
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # 创建黑色背景
        seg = mask * 0


        # 填充轮廓内部
        cv2.drawContours(
            seg,
            contours,
            -1,
            255,
            thickness=-1
        )


        # 保存名字转换

        new_name = (
            name
            .replace(
                "_Annotation",
                "_seg"
            )
        )


        cv2.imwrite(
            str(label_dir / new_name),
            seg
        )


        count += 1


print(
    "数据处理完成！"
)

print(
    f"生成 {count} 个 mask"
)