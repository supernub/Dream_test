# 下载 CellFM 预训练权重

## 方法 1: 使用下载脚本（推荐）

```bash
cd /home/ubuntu/LLM-inference/xinze-project/dream_test
./download_cellfm_checkpoint.sh
```

这个脚本会自动从 HuggingFace 下载权重文件。

## 方法 2: 使用 git lfs clone（从 HuggingFace）

```bash
cd /home/ubuntu/LLM-inference/xinze-project/cellfm

# 如果文件已经在其他位置，直接复制
cp /path/to/CellFM_80M_weight.ckpt ./

# 或者从 HuggingFace clone (需要 git lfs)
# git lfs install
# git clone https://huggingface.co/ShangguanNingyuan/CellFM.git
```

## 方法 3: 直接下载文件

```bash
cd /home/ubuntu/LLM-inference/xinze-project/cellfm

# 使用 wget
wget https://huggingface.co/ShangguanNingyuan/CellFM/resolve/main/CellFM_80M_weight.ckpt

# 或使用 curl
curl -L https://huggingface.co/ShangguanNingyuan/CellFM/resolve/main/CellFM_80M_weight.ckpt -o CellFM_80M_weight.ckpt
```

## 文件信息

- **文件名**: `CellFM_80M_weight.ckpt`
- **大小**: ~320 MB
- **位置**: `/home/ubuntu/LLM-inference/xinze-project/cellfm/`
- **来源**: HuggingFace Model Hub (ShangguanNingyuan/CellFM)

## 验证下载

下载完成后验证：

```bash
ls -lh /home/ubuntu/LLM-inference/xinze-project/cellfm/CellFM_80M_weight.ckpt
# 应该显示 ~320 MB 的文件
```

## 如果有现成的权重文件

如果你已经有一份 checkpoint 文件（无论是在服务器上的其他位置，还是你已经下载好的），可以：

```bash
# 找到文件位置
find /home/ubuntu -name "CellFM*.ckpt" -type f

# 或查看你的位置
ls -lh /path/to/your/checkpoint.ckpt

# 复制到 cellfm 目录
cp /path/to/your/CellFM_80M_weight.ckpt /home/ubuntu/LLM-inference/xinze-project/cellfm/
```




