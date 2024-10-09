# Hindi OCR Model Training and Inference

This repository contains code for training and inferring a Hindi Optical Character Recognition (OCR) model using the `GOT-OCR2.0` model. It leverages Kaggle's environment for GPU processing and works with Hindi synthetic line image-text pairs for model training.

## Table of Contents

- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Training](#training)
- [Inference](#inference)
- [Checkpoint Management](#checkpoint-management)
- [Sample Data Format](#sample-data-format)
- [License](#license)

## Installation

1. Clone the `ms-swift` repository and navigate to its directory:

    ```bash
    !git clone https://github.com/modelscope/ms-swift.git
    %cd ms-swift
    ```

2. Install the dependencies required for LLM training:

    ```bash
    !pip install -e .[llm]
    ```

3. Install additional libraries like `verovio` for music score processing (if needed):

    ```bash
    !pip install verovio
    ```

## Dataset Structure

The dataset used for this project includes images and corresponding transcriptions stored in a CSV file. The structure is as follows:

- `image_base_path`: Directory containing the synthetic Hindi OCR line images.
- `CSV file`: Contains image filenames and their respective text transcriptions.

Example of a row in the CSV:

| image_file  | text          |
|-------------|---------------|
| image_01.png | यह एक उदाहरण है |

## Training

To train the OCR model, follow these steps:

1. Prepare the dataset by converting it to JSON format compatible with the `GOT-OCR2.0` model:

    ```python
    import os
    import json

    image_base_path = '/kaggle/input/hindi-ocr-synthetic-line-image-text-pair/data_80k/output_images/'
    json_data = []

    for index, row in df.iterrows():
        full_image_path = os.path.join(image_base_path, row['image_file'])
        json_obj = {
            "query": "<image>Transcribe the text in this image",
            "response": row['text'],
            "images": [full_image_path]
        }
        json_data.append(json_obj)

    json_output = json.dumps(json_data, indent=4, ensure_ascii=False)
    with open('/kaggle/working/output_data.json', 'w', encoding='utf-8') as f:
        f.write(json_output)
    ```

2. Begin the training process using the following command:

    ```bash
    !swift sft \
    --model_type got-ocr2 \
    --model_id_or_path stepfun-ai/GOT-OCR2_0 \
    --sft_type lora \
    --dataset /kaggle/working/output_data.json \
    --output_dir /kaggle/working/hindi_got_model_3 \
    --num_train_epochs 1 \
    --max_steps 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 200
    ```

3. The training checkpoint will be saved in the specified `output_dir`.

## Inference

1. Create a JSON file for testing, containing a subset of images for inference:

    ```python
    import os
    import json

    image_base_path = '/kaggle/input/hindi-ocr-synthetic-line-image-text-pair/data_80k/TestSamples/'
    json_data = []

    for index, row in df.iterrows():
        if index >= 10:  # Process only 10 test files
            break
        full_image_path = os.path.join(image_base_path, row['image_file'])
        json_obj = {
            "query": "<image>Transcribe the text in this image",
            "response": row['text'],
            "images": [full_image_path]
        }
        json_data.append(json_obj)

    json_output = json.dumps(json_data, indent=4, ensure_ascii=False)
    with open('/kaggle/working/test1.json', 'w', encoding='utf-8') as f:
        f.write(json_output)
    ```

2. Run inference using the trained checkpoint:

    ```bash
    !CUDA_VISIBLE_DEVICES=0 swift infer\
    --ckpt_dir /kaggle/working/hindi_got_model_3/got-ocr2/v0-20240930-060444/checkpoint-1000 \
    --dataset /kaggle/working/test1.json \
    --load_dataset_config true
    ```

## Checkpoint Management

1. After training, compress the checkpoint for easy downloading or transfer:

    ```bash
    !zip -r /kaggle/working/checkpoint-1000.zip /kaggle/working/hindi_got_model_3/got-ocr2/v0-20240930-060444/checkpoint-1000
    ```

The zip file will be created in the `/kaggle/working/` directory.

## Sample Data Format

The dataset follows the format below for storing images and transcriptions. Here is an example structure:

```json
[
    {
        "query": "<image>Transcribe the text in this image",
        "response": "गर्भनिरोध के लिए महिलाएं क्यों कराती हैं नसबंदी",
        "images": [
            "F:/archive (3)/data_80k/TestSamples/1.png"
        ]
    },
    {
        "query": "<image>Transcribe the text in this image",
        "response": "'मस्‍ज‍िद ख़ुदा का घर है तो यह ईमान वाली स्‍त्र‍ियों के लिए कैसे बंद हो सकता है'",
        "images": [
            "F:/archive (3)/data_80k/TestSamples/2.png"
        ]
    },
    {
        "query": "<image>Transcribe the text in this image",
        "response": "नज़रिया: गोरखपुर, नागपुर और दिल्ली के त्रिकोण में फंसा है 2019",
        "images": [
            "F:/archive (3)/data_80k/TestSamples/3.png"
        ]
    },
    ...
    {
        "query": "<image>Transcribe the text in this image",
        "response": "जनता की ईमानदारी, पुलिस की मुसीबत",
        "images": [
            "/kaggle/input/hindi-ocr-synthetic-line-image-text-pair/data_80k/TestSamples/11.png"
        ]
    }
]
