import datasets
import json
import os
from PIL import Image
from tqdm import tqdm
import argparse
import multiprocessing as mp

NUM_START = 2

def preprocess_image(example, output_dir):
    """
    データセットの例から画像をロードし、処理する。
    """

    source = example['source'].replace('\n\n', '').strip()
    image_path = example['path']
    extracted_info = []

    image_full_path = None
    if source == 'coco' or source == 'llava' or source == 'journeydb':
        image_full_path = os.path.join("./dataset/row", image_path)
    elif source == 'sam':
        target_file = os.path.basename(image_path)
        image_full_path = os.path.join(f"./dataset/row/sam/extract_{NUM_START}", target_file)

    if image_full_path:
        try:
            with Image.open(image_full_path) as img:
                save_path = os.path.join(output_dir, image_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if not os.path.exists(save_path):
                    img.save(save_path)

                    extracted_info.append({
                        "file_name": os.path.basename(image_path),
                        "source": source
                    })

        except (FileNotFoundError, Exception) as e:
            print(f"画像の処理中にエラーが発生しました ({image_full_path}): {e}")
            pass

    return extracted_info

def extract_and_save_images(caption_to_find, output_json_path):
    try:
        train_dataset = datasets.load_dataset("luping-liu/LongAlign", split="train", num_proc=8)
    except Exception as e:
        print(f"データセットのロード中にエラーが発生しました: {e}")
        return

    if caption_to_find == 'sam':
        output_dir = f"./dataset/sam_{NUM_START}"
    else:
        output_dir = "./dataset"

    extracted_images_info = []
    print(f"データセットからキャプション '{caption_to_find}' に一致する画像を検索中...")

    # データセットをフィルタリング
    filtered_dataset = train_dataset.filter(lambda example: example['source'].strip() == caption_to_find and 100 <= int(example['path'].split('/')[2]) <= 199)
    total_samples = len(filtered_dataset)

    # 直列処理
    results = []
    for example in tqdm(filtered_dataset, total=total_samples, desc="Processing images serially"):
        results.append(preprocess_image(example, output_dir))

    # 結果をまとめる
    for res in results:
        extracted_images_info.extend(res)

    # 抽出した情報をJSONファイルに保存
    if os.path.exists(output_json_path):
        # 既存のデータを読み込む
        with open(output_json_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.extend(extracted_images_info)

    with open(output_json_path, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"\n合計 {len(extracted_images_info)} 件の画像情報が {output_json_path} に保存されました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save images from a dataset.')
    parser.add_argument('--source', type=str, required=True, help='The name of the dataset source to process (e.g., coco, sam, llava, journeydb).')
    args = parser.parse_args()

    target_source = args.source

    output_file = f"{target_source}_extracted_images.json"
    print(output_file)

    extract_and_save_images(target_source, output_file)
