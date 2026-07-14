# Social Media Depression Detection

This repository provides the implementation of a social media depression detection framework evaluated on the eRisk 2017 and eRisk 2018 datasets.

## Repository Structure

```text
.
├── proc_pipeline/              # Data preprocessing scripts
├── processed_2017/             # Processed eRisk 2017 data
├── processed_2018/             # Processed eRisk 2018 data
├── pretrain_file/              # Pretrained model files and download URLs
├── output_file/                # Experimental outputs
├── main_hier_clf.py            # Hierarchical classification experiment
├── main_post_clf.py            # Post-level classification experiment
├── E2_LPS_raw_simplified_2gpu_paper_correct.py
└── 2-shot experiment.py        # Two-shot experiment
```

## Requirements

The experiments were conducted using Python 3.8 and CUDA 12.1.

The main dependencies are:

```text
torch==2.1.0+cu121
torchaudio==2.1.0+cu121
torchmetrics==1.5.2
torchvision==0.16.0+cu121
tqdm==4.67.1
transformers==4.46.3
sentence-transformers==3.2.1
sentencepiece==0.2.0
numpy==1.24.3
```

The dependencies can be installed using:

```bash
pip install torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install torchmetrics==1.5.2 \
    tqdm==4.67.1 \
    transformers==4.46.3 \
    sentence-transformers==3.2.1 \
    sentencepiece==0.2.0 \
    numpy==1.24.3
```

## Dataset

We cannot include the datasets in this repository due to their terms of use and distribution restrictions.

Please request access to the datasets through the official eRisk websites:

* eRisk 2017: https://erisk.irlab.org/2017/index.html
* eRisk 2018: https://erisk.irlab.org/2018/index.html

After obtaining the datasets, place the downloaded files in the corresponding directories:

```text
processed_2017/
processed_2018/
```

Please comply with the terms, licenses, and restrictions specified by the dataset providers.

## Data Preprocessing

The data preprocessing scripts are provided in the `proc_pipeline/` directory.

Enter the directory:

```bash
cd proc_pipeline
```

Run the preprocessing script corresponding to the dataset being used.

The processed eRisk 2017 and eRisk 2018 data should be saved in:

```text
processed_2017/
processed_2018/
```

The exact input and output paths should be configured according to the local directory structure.

## Pretrained Models

The experiments require several pretrained models.

The download URLs for the required models are provided in the `pretrain_file/` directory. Download the models and place them in the corresponding subdirectories.

The directory should contain files similar to:

```text
pretrain_file/
├── bert_base_uncased/
├── all-MiniLM-L6-v2/
└── ...
```

Before running the experiments, ensure that the paths specified by `--model_type`, `--screen_model`, and `--detect_model` point to the correct local model directories.

## Experiments

Replace the placeholder paths in the following commands with the actual local paths.

### 1. Hierarchical Classification

Run the hierarchical classification experiment using:

```bash
python -u main_hier_clf.py \
    --lr=2e-5 \
    --threshold=0.5 \
    --input_dir=process_data_path \
    --bs=4 \
    --model_type=../pretrain_file/bert_base_uncased
```

Here, `process_data_path` should be replaced with the path to the processed eRisk dataset.

### 2. Post-Level Classification

Run the post-level classification experiment using:

```bash
python -u main_post_clf.py \
    --lr=2e-5 \
    --threshold=0.5 \
    --input_dir=process_data_path \
    --bs=4 \
    --model_type=../pretrain_file/bert_base_uncased
```

### 3. E2-LPS Experiment

The E2-LPS experiment uses separate devices for post screening and depression detection.

Run:

```bash
python -u E2_LPS_raw_simplified_2gpu_paper_correct.py \
    --data_root=processed_data_path \
    --screen_model=../pretrain_file/all-MiniLM-L6-v2 \
    --detect_model=../pretrain_file/bert_base_uncased \
    --output_dir=output_file \
    --epochs=10 \
    --top_ratio=0.125 \
    --max_posts_per_user=0 \
    --screen_max_len=128 \
    --detect_max_len=128 \
    --screen_device=cuda:0 \
    --detect_device=cuda:1 \
    --gradient_checkpointing \
    --infer_selected_only \
    --export_screened
```

The main arguments are:

* `--data_root`: path to the processed dataset.
* `--screen_model`: path to the sentence-transformer model used for post screening.
* `--detect_model`: path to the pretrained BERT model used for depression detection.
* `--output_dir`: directory used to save predictions, selected posts, logs, and evaluation results.
* `--epochs`: number of training epochs.
* `--top_ratio`: proportion of posts retained by the screening module.
* `--max_posts_per_user=0`: use all available posts for each user.
* `--screen_device`: GPU used for the screening module.
* `--detect_device`: GPU used for the detection module.
* `--export_screened`: export the selected posts after screening.

This command assumes that two CUDA-enabled GPUs are available. Device settings may be adjusted according to the local hardware environment.

### 4. Two-Shot Experiment

Run the two-shot experiment using:

```bash
python "2-shot experiment.py"
```

Quotation marks are required because the filename contains a space.

## License

This repository is intended for academic research purposes only.

Users must comply with the licenses, data-use agreements, and ethical requirements of the eRisk datasets and the pretrained models used in this project.
