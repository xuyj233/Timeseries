# 评估阶段Bug修复总结

## 问题描述

用户在运行评估脚本时遇到了两个关键错误：
1. `ValueError: too many values to unpack (expected 2)`
2. `RuntimeError: maximum size for tensor at dimension 2 is 7 but size is 96`

## 根本原因

问题的根源在于**训练数据和评估数据的维度不匹配**：

- **训练时（S3数据）**: 使用单变量时间序列
  - 数据形状: `(batch_size, seq_length)` - **2D张量**
  - S3Preprocessor会将多变量序列拆分成多个单变量序列

- **评估时（标准数据集，如ETTH1）**: 使用多变量时间序列
  - 数据形状: `(batch_size, seq_length, n_features)` - **3D张量**
  - ETTH1有7个特征，所以最后一维大小为7

Timer模型的原始实现假设输入是2D张量（单变量），导致在处理3D张量（多变量）时出错。

## 修复详情

### 1. 修复 `models/timer_model.py` - 第244行

**问题**: 代码假设`input_ids`是2D张量
```python
batch_size, seq_length = input_ids.shape  # 错误：input_ids可能是3D的
```

**修复**: 添加维度检查
```python
if input_ids.ndim == 2:
    batch_size, seq_length = input_ids.shape
else:
    batch_size, seq_length = input_ids.shape[:2]
```

### 2. 修复 `models/timer_model.py` - TimerPatchEmbedding类

**问题**: 
- Patch embedding在**特征维度**（dimension=-1）上进行unfold
- 对于3D输入 `(batch, seq_len, 7)`，最后一维是7
- 但unfold需要size=96（input_token_len），远大于7，导致错误

**原始代码**:
```python
def forward(self, hidden_state: torch.Tensor):
    hidden_state = hidden_state.unfold(
        dimension=-1, size=self.input_token_len, step=self.input_token_len)
    return self.emb(hidden_state)
```

**修复方案**: 正确处理多变量输入
```python
def forward(self, hidden_state: torch.Tensor):
    if hidden_state.ndim == 2:
        # 单变量: (batch_size, seq_length)
        hidden_state = hidden_state.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
    
    elif hidden_state.ndim == 3:
        # 多变量: (batch_size, seq_length, n_features)
        # 在时间维度上进行patching
        
        # 1. 转置: (batch_size, n_features, seq_length)
        hidden_state = hidden_state.transpose(1, 2)
        
        # 2. 在时间维度上unfold: (batch, n_features, num_patches, patch_len)
        hidden_state = hidden_state.unfold(
            dimension=-1, size=self.input_token_len, step=self.input_token_len)
        
        # 3. 在特征维度上平均，得到: (batch, num_patches, patch_len)
        batch_size, n_features, num_patches, patch_len = hidden_state.shape
        hidden_state = hidden_state.permute(0, 2, 1, 3)
        hidden_state = hidden_state.view(batch_size, num_patches, n_features, patch_len).mean(dim=2)
    
    return self.emb(hidden_state)
```

### 3. 已修复的其他问题

在之前的修复中，我们还解决了：

#### A. JSON序列化错误
**问题**: `TypeError: Object of type float32 is not JSON serializable`

**修复**: 在 `scripts/evaluate.py` 中添加自定义JSON编码器
```python
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
```

#### B. 数据加载问题
**问题**: DataLoader返回的batch数据解包

**修复**: 在 `scripts/evaluate.py` 中正确处理batch数据
```python
for batch_idx, batch_data in enumerate(test_loader):
    if isinstance(batch_data, (tuple, list)):
        if len(batch_data) == 2:
            history, target = batch_data
        elif len(batch_data) > 2:
            history, target = batch_data[0], batch_data[1]
```

## 验证结果

修复后，评估脚本成功运行：

```bash
python scripts\evaluate.py --model-path outputs\quick_test\best_model --datasets ETTH1 --batch-size 32 --output-dir outputs\quick_eval_v2
```

**输出**:
```
Test samples: 2717
  Processed 10/85 batches
  Processed 20/85 batches
  ...
  Processed 80/85 batches

ETTH1 Results:
  MSE:  1.319881
  MAE:  0.887022
  RMSE: 1.148861
  MAPE: 232.38%
  Direction Acc: 47.20%

[OK] Results saved to outputs\quick_eval_v2\evaluation_results.json
```

✅ **所有批次处理成功，无错误！**

## 关于模型性能

目前的评估性能不佳（MAPE=232.38%），这是**预期的**，原因如下：

1. **训练/评估数据不匹配**:
   - 训练: S3单变量数据（从UTSD数据集提取）
   - 评估: ETTH1多变量数据（7个特征）

2. **需要微调（Fine-tuning）**:
   - 预训练模型学习的是通用时间序列模式
   - 在特定数据集上需要微调才能获得好的性能

3. **架构适配**:
   - 当前的多变量处理方案（在特征维度上平均）是简化的
   - 更好的方案可能包括：
     - Channel-Independent策略（每个特征独立处理）
     - 学习特征权重而不是简单平均
     - 使用注意力机制融合多个特征

## 后续建议

1. **Fine-tuning**: 在ETTH1等标准数据集上微调预训练模型
2. **改进多变量处理**: 考虑更复杂的多变量融合策略
3. **数据对齐**: 或者在评估时也使用单变量数据（每个特征独立评估）

## 修改的文件

1. `models/timer_model.py`
   - 修复input_ids形状解包（第244-248行）
   - 重写TimerPatchEmbedding.forward方法（第34-73行）

2. `scripts/evaluate.py`
   - 添加NpEncoder类（已在之前修复）
   - 改进错误处理和traceback输出

## 结论

所有评估阶段的关键bug已修复，评估脚本现在可以正常运行并生成结果。核心问题是Timer模型需要同时支持2D（单变量）和3D（多变量）输入张量，现在已经实现了这种兼容性。




