from pathlib import Path
import time
import zipfile

import requests


# =========================================================
# 1. 数据集下载地址
# =========================================================

url = "https://zenodo.org/records/1322001/files/training_set.zip?download=1"


# =========================================================
# 2. 保存路径
# =========================================================

data_dir = Path("data")
zip_path = data_dir / "training_set.zip"
extract_dir = data_dir / "HC18"

data_dir.mkdir(exist_ok=True)


# =========================================================
# 3. 支持断点续传的下载函数
# =========================================================

def download_file(url, save_path, max_retries=10):

    for attempt in range(max_retries):

        # 看看之前已经下载了多少
        downloaded = save_path.stat().st_size if save_path.exists() else 0

        headers = {}

        if downloaded > 0:
            headers["Range"] = f"bytes={downloaded}-"

            print(
                f"\n检测到已有文件："
                f"{downloaded / 1024 / 1024:.1f} MB"
            )

            print("尝试从断点继续下载...")

        try:

            response = requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(15, 60)
            )

            response.raise_for_status()

            # -------------------------------------------------
            # 如果服务器接受断点续传，一般返回 206
            # -------------------------------------------------

            if downloaded > 0 and response.status_code == 206:

                mode = "ab"

                remaining = int(
                    response.headers.get(
                        "content-length",
                        0
                    )
                )

                total_size = downloaded + remaining

            else:

                # 服务器没有接受 Range
                # 那就重新下载
                mode = "wb"
                downloaded = 0

                total_size = int(
                    response.headers.get(
                        "content-length",
                        0
                    )
                )

            # -------------------------------------------------
            # 开始写文件
            # -------------------------------------------------

            with open(save_path, mode) as f:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if not chunk:
                        continue

                    f.write(chunk)

                    downloaded += len(chunk)

                    if total_size > 0:

                        percent = (
                            downloaded
                            / total_size
                            * 100
                        )

                        print(
                            f"\r下载进度："
                            f"{percent:6.2f}% "
                            f"({downloaded / 1024 / 1024:.1f} MB"
                            f"/"
                            f"{total_size / 1024 / 1024:.1f} MB)",
                            end=""
                        )

            print("\n下载完成！")

            return

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.Timeout
        ) as e:

            print("\n\n下载连接中断：")
            print(e)

            print(
                f"\n第 {attempt + 1}/{max_retries} 次失败，"
                "5 秒后继续..."
            )

            time.sleep(5)

    raise RuntimeError("多次尝试后仍然无法完成下载。")


# =========================================================
# 4. 下载
# =========================================================

print("开始下载 HC18 training_set.zip ...")

download_file(url, zip_path)


# =========================================================
# 5. 检查并解压
# =========================================================

print("\n正在检查 ZIP 文件...")

if not zipfile.is_zipfile(zip_path):
    raise RuntimeError(
        "下载得到的文件不是完整 ZIP，"
        "请检查网络后重新下载。"
    )

print("ZIP 文件完整。")
print("开始解压...")


with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)


print("解压完成！")
print("数据位置：", extract_dir.resolve())