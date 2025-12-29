# GPU 使用说明

## ✅ 默认设置

**脚本已默认使用第0张GPU卡**

## 🚀 使用方法

### 方法1：使用 run_pipeline.sh 脚本参数（推荐）

```bash
# 使用第0张卡（默认）
./run_pipeline.sh

# 使用第1张卡
./run_pipeline.sh 1

# 使用第2张卡
./run_pipeline.sh 2

# 使用第3张卡
./run_pipeline.sh 3
```

### 方法2：直接设置环境变量

```bash
# 使用第0张卡
CUDA_VISIBLE_DEVICES=0 python3 arrange_hallucination_data.py

# 使用第1张卡
CUDA_VISIBLE_DEVICES=1 python3 arrange_hallucination_data.py

# 使用第2张卡
CUDA_VISIBLE_DEVICES=2 python3 arrange_hallucination_data.py

# 使用多张卡
CUDA_VISIBLE_DEVICES=0,1 python3 arrange_hallucination_data.py
```

### 方法3：在脚本运行前设置

```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 然后运行脚本
./run_pipeline.sh

# 或
python3 arrange_hallucination_data.py
```

## 🔍 检查GPU使用情况

### 查看可用GPU

```bash
# 查看所有GPU
nvidia-smi

# 持续监控GPU使用
watch -n 1 nvidia-smi
```

### 查看脚本使用的GPU

运行脚本时会显示：

```
[INFO] 当前使用GPU: 0
使用GPU: 0
设备: cuda:0
```

## 📊 多GPU环境建议

如果有多张GPU卡，建议：

### 1. 查看GPU空闲情况

```bash
nvidia-smi
```

### 2. 选择空闲的GPU

```bash
# 例如GPU 2 是空闲的
./run_pipeline.sh 2
```

### 3. 并行处理（高级）

如果想同时处理多个数据集：

```bash
# 终端1: 使用GPU 0处理训练集
CUDA_VISIBLE_DEVICES=0 python3 process_train.py &

# 终端2: 使用GPU 1处理测试集
CUDA_VISIBLE_DEVICES=1 python3 process_test.py &
```

## ⚙️ 优先级说明

GPU设置的优先级（从高到低）：

1. **运行时环境变量**
   ```bash
   CUDA_VISIBLE_DEVICES=1 ./run_pipeline.sh
   ```

2. **脚本参数**
   ```bash
   ./run_pipeline.sh 1
   ```

3. **脚本默认值**
   ```bash
   ./run_pipeline.sh  # 使用GPU 0
   ```

## 🐛 常见问题

### Q1: 如何确认脚本正在使用正确的GPU？

**方法1**: 查看脚本输出

```
使用GPU: 0
设备: cuda:0
```

**方法2**: 使用nvidia-smi

```bash
# 在另一个终端
watch -n 1 nvidia-smi
# 查看哪个GPU的显存在增加
```

### Q2: 出现"CUDA out of memory"错误

**解决方案**:

1. 换一张显存更大的GPU
   ```bash
   ./run_pipeline.sh 1  # 换到GPU 1
   ```

2. 或减小批次大小（在arrange_hallucination_data.py中）
   ```python
   batch_size = 32  # 从128减少
   ```

### Q3: 如何使用CPU处理？

```bash
# 设置为空或-1
CUDA_VISIBLE_DEVICES="" python3 arrange_hallucination_data.py

# 或
CUDA_VISIBLE_DEVICES=-1 python3 arrange_hallucination_data.py
```

**注意**: CPU处理会非常慢（10-20倍）

## 📋 快速参考

| 需求 | 命令 |
|------|------|
| 使用GPU 0（默认） | `./run_pipeline.sh` |
| 使用GPU 1 | `./run_pipeline.sh 1` |
| 使用GPU 2 | `./run_pipeline.sh 2` |
| 查看GPU使用 | `nvidia-smi` |
| 监控GPU | `watch -n 1 nvidia-smi` |
| 使用CPU | `CUDA_VISIBLE_DEVICES="" ./run_pipeline.sh` |

## ✨ 最佳实践

1. **运行前检查GPU**
   ```bash
   nvidia-smi
   ```

2. **选择空闲GPU**
   ```bash
   # 假设GPU 1是空闲的
   ./run_pipeline.sh 1
   ```

3. **监控处理过程**
   ```bash
   # 在新终端
   watch -n 1 nvidia-smi
   ```

4. **后台运行时指定GPU**
   ```bash
   nohup ./run_pipeline.sh 1 > process.log 2>&1 &
   ```

---

**默认配置**: 使用第0张GPU卡

**推荐**: 运行前用 `nvidia-smi` 查看GPU使用情况，选择空闲的卡
