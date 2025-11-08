本仓库实现了一个简化版的 Transformer，包括：

- Encoder-only 模型：用于 AG News 文本分类；
- Decoder-only 模型：在 AG News 语料上做自回归语言建模。

本文档主要说明：

1. 环境与依赖；
2. 推荐硬件配置；
3. 如何用 exact 命令行（含随机种子） 复现实验结果。


## 1. 环境与依赖

使用 Conda 创建独立环境（Python 3.10）：

```bash
conda create -n llm python=3.10
conda activate llm
````

安装依赖（简化示例）：

```bash
pip install torch torchvision torchaudio
pip install datasets transformers matplotlib
```

> 说明：
>
> * 代码会通过 Hugging Face `datasets` 自动下载 `ag_news` 数据集；
> * 如果需要使用镜像或离线缓存，请在运行前自行配置环境变量（例如 `HF_ENDPOINT` 等）。


## 2. 硬件要求

* 建议配置：**单卡 NVIDIA GPU，显存 ≥ 12GB**
  实际实验环境示例：

  * GPU：NVIDIA GeForce RTX 3090（24GB）
  * GPU 数量：1 张
  * CPU：常规多核 CPU 即可
  * 内存：≥ 16GB


## 3. 目录结构（简要）

核心代码文件：

* `src/models.py`：Transformer 模块（多头注意力、Encoder/Decoder 层、分类头、LM 头等）
* `src/utils.py`：工具函数（种子设置、mask 构造、绘图、日志等）
* `src/train_cls.py`：AG News 文本分类训练脚本（Encoder-only）
* `src/train_lm_agnews.py`：AG News 语言建模训练脚本（Decoder-only）
* `scripts/run.sh`：封装好的运行脚本（部分实验）


## 4. 复现实验：Encoder-only 文本分类（AG News）

### 4.1 基线模型（2 层 Encoder，4 头）

**实验设置：**

* `d_model = 128`
* `d_ff = 512`
* `heads = 4`
* `layers = 2`
* `dropout = 0.1`
* `batch_size = 64`
* `lr = 3e-4`
* `epochs = 5`
* `max_len = 256`
* `seed = 42`

**命令行：**

```bash
python -m src.train_cls \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 4 \
  --layers 2 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 5 \
  --seed 42 \
  --max_len 256 \
  --save_dir results/agnews_encoder
```


```bash
bash scripts/run.sh cls
```

> `scripts/run.sh` 中的 `cls` 分支默认就是上述配置（含 `--seed 42`），保存结果到 `results/agnews_encoder`。


### 4.2 注意力头数消融（Heads Ablation）

**实验设置：**

* 固定：
  * `d_model = 128`
  * `d_ff = 512`
  * `layers = 2`
  * 其余超参数同基线
* 只改变 `heads ∈ {2, 4, 8}`

**命令行：**

```bash
# 2 头
python -m src.train_cls \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 2 \
  --layers 2 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 5 \
  --seed 42 \
  --max_len 256 \
  --save_dir results/cls_H2

# 4 头
python -m src.train_cls \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 4 \
  --layers 2 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 5 \
  --seed 42 \
  --max_len 256 \
  --save_dir results/cls_H4

# 8 头
python -m src.train_cls \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 8 \
  --layers 2 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 5 \
  --seed 42 \
  --max_len 256 \
  --save_dir results/cls_H8
```


### 4.3 网络层数消融（Layers Ablation）

**实验设置：**

* 固定：

  * `d_model = 128`
  * `d_ff = 512`
  * `heads = 4`
* 只改变 `layers ∈ {1, 2, 4}`

**命令行：**

```bash
# 1 层 Encoder
python -m src.train_cls \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 4 \
  --layers 1 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 5 \
  --seed 42 \
  --max_len 256 \
  --save_dir results/cls_L1

# 2 层 Encoder（与基线一致）
python -m src.train_cls \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 4 \
  --layers 2 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 5 \
  --seed 42 \
  --max_len 256 \
  --save_dir results/cls_L2

# 4 层 Encoder
python -m src.train_cls \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 4 \
  --layers 4 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 5 \
  --seed 42 \
  --max_len 256 \
  --save_dir results/cls_L4
```


### 4.4 位置编码消融（Positional Encoding On / Off）

本实验比较 **使用位置编码** 与 **去掉位置编码** 对 AG News 分类性能的影响。

1. **位置编码开启（PE on）**
   使用原始实现（在 `PositionalEncoding.forward` 中有 `x = x + self.pe[:, :L, :]`），例如：

   ```bash
   python -m src.train_cls \
     --model_dim 128 \
     --ffn_dim 512 \
     --heads 4 \
     --layers 4 \
     --dropout 0.1 \
     --batch_size 64 \
     --lr 3e-4 \
     --epochs 5 \
     --seed 42 \
     --max_len 256 \
     --save_dir results/cls_PE_on
   ```

2. **位置编码关闭（PE off）**
   在 `src/models.py` 中（或对应文件）将 `PositionalEncoding` 的 `forward` 改为不加上 `self.pe`，例如：

   ```python
   def forward(self, x):
       L = x.size(1)
       # 原版：x = x + self.pe[:, :L, :]
       return self.dropout(x)
   ```

   修改后重新运行：

   ```bash
   python -m src.train_cls \
     --model_dim 128 \
     --ffn_dim 512 \
     --heads 4 \
     --layers 4 \
     --dropout 0.1 \
     --batch_size 64 \
     --lr 3e-4 \
     --epochs 5 \
     --seed 42 \
     --max_len 256 \
     --save_dir results/cls_PE_off
   ```

> 注意：PE on / off 的差异来自代码实现本身，而不是命令行参数，所以需要在运行前确认 `PositionalEncoding` 的实现状态。


## 5. 复现实验：Decoder-only 语言建模（AG News）

### 5.1 实验设置

* 任务：在 AG News 文本上做自回归语言建模（忽略标签）
* 模型结构：

  * Decoder-only Transformer
  * `d_model = 128`
  * `d_ff = 512`
  * `heads = 4`
  * `layers = 4`
  * `dropout = 0.1`
* 训练设置：

  * `batch_size = 64`
  * `lr = 3e-4`
  * `epochs = 10`
  * `max_len = 256`
  * `tokenizer_name = bert-base-uncased`
  * `seed = 42`

### 5.2 命令行

```bash
python -m src.train_lm_agnews \
  --model_dim 128 \
  --ffn_dim 512 \
  --heads 4 \
  --layers 4 \
  --dropout 0.1 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 10 \
  --max_len 256 \
  --save_dir results/lm_agnews \
  --tokenizer_name bert-base-uncased \
  --seed 42
```

运行完成后，`results/lm_agnews` 目录中会包含：

* 训练与验证的 loss / perplexity 日志（例如 `.jsonl` 或 `.txt`）；
* 模型权重文件（如 `best.pt` 或类似命名）；
* 可能的曲线图（如 `loss_curve.png`、`val_ppl.png` 等，视实现而定）。
