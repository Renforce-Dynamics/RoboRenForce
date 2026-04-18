# Environment Setup with UV

**推荐方法**: 使用uv管理Python环境和依赖

---

## 🚀 快速开始

### 1. 安装uv（如果还没有）

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用pip
pip install uv
```

### 2. 创建并激活虚拟环境

```bash
# 进入项目目录
cd /home/ununtu/code/RoboRenforce

# 创建虚拟环境并安装所有依赖
uv sync

# 激活虚拟环境
source .venv/bin/activate
```

### 3. 验证安装

```bash
# 测试导入
python -c "import torch; import transformers; print('✓ Environment ready')"

# 测试LeRobotDataset
python -c "
import sys
sys.path.insert(0, 'source/RoboRenForce')
from RoboRenForce.dataset.lerobot import LeRobotDatasetCfg
print('✓ LeRobotDataset imported')
"
```

---

## 📦 依赖管理

### 安装核心依赖

```bash
# 所有核心依赖已在pyproject.toml中定义
uv sync
```

### 安装可选依赖

```bash
# LoRA fine-tuning支持
uv sync --extra lora

# 开发工具
uv sync --extra dev

# 所有可选依赖
uv sync --extra all
```

### 添加新依赖

```bash
# 添加到核心依赖
uv add package-name

# 添加到开发依赖
uv add --dev package-name

# 添加到可选依赖组
uv add --optional lora package-name
```

### 更新依赖

```bash
# 更新所有依赖
uv sync --upgrade

# 更新特定包
uv add package-name --upgrade
```

---

## 🔧 常用命令

### 环境管理

```bash
# 创建/同步环境
uv sync

# 激活环境
source .venv/bin/activate

# 停用环境
deactivate

# 删除环境
rm -rf .venv
```

### 包管理

```bash
# 安装包
uv add package-name

# 移除包
uv remove package-name

# 列出已安装的包
uv pip list

# 检查过时的包
uv pip list --outdated
```

### 运行脚本

```bash
# 在虚拟环境中运行Python
uv run python script.py

# 在虚拟环境中运行任意命令
uv run pytest tests/
```

---

## 🆚 对比Conda

| 特性 | uv | conda |
|------|-----|-------|
| 速度 | ⚡ 极快（Rust实现） | 🐢 较慢 |
| 内存占用 | 💾 小 | 💾 大 |
| 虚拟环境大小 | 📦 小（符号链接） | 📦 大（完整复制） |
| 依赖解析 | ✅ 快速准确 | ⚠️ 较慢 |
| 兼容性 | ✅ 标准PyPI | ✅ Conda+PyPI |

### 从Conda迁移

如果你之前使用conda：

```bash
# 1. 停用conda环境
conda deactivate

# 2. 使用uv创建新环境
uv sync

# 3. 激活uv环境
source .venv/bin/activate

# 4. (可选) 删除conda环境
conda env remove -n RRF
```

---

## 🐛 常见问题

### Q: uv sync失败怎么办？

```bash
# 清理缓存并重试
rm -rf .venv uv.lock
uv sync
```

### Q: 如何指定Python版本？

```bash
# 编辑.python-version文件
echo "3.10" > .python-version

# 重新创建环境
uv sync
```

### Q: CUDA/PyTorch版本问题？

```bash
# 指定PyTorch版本和CUDA版本
uv add "torch==2.1.0+cu121" --index-url https://download.pytorch.org/whl/cu121
```

### Q: 如何在不同机器间共享环境？

```bash
# 导出lock文件（已自动生成）
# 其他机器上直接运行
uv sync
```

---

## 📋 项目依赖清单

### 核心依赖
- **torch**: 深度学习框架
- **transformers**: HuggingFace模型库
- **pandas/pyarrow**: 数据处理
- **safetensors**: 模型存储
- **einops**: 张量操作

### 可选依赖
- **peft**: LoRA fine-tuning
- **pytest**: 测试框架
- **black/isort**: 代码格式化

完整依赖见：[pyproject.toml](pyproject.toml)

---

## ✅ 验证清单

安装完成后，确保以下都能正常工作：

```bash
# 1. Python版本
python --version  # 应该是3.10+

# 2. PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 3. Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 4. 项目导入
python -c "import sys; sys.path.insert(0, 'source/RoboRenForce'); from RoboRenForce.dataset.lerobot import LeRobotDatasetCfg; print('✓ OK')"
```

---

## 🚀 下一步

环境配置完成后：

1. 查看快速开始: [QUICKSTART.md](QUICKSTART.md)
2. 查看TODO清单: [TODO-NEXT.md](TODO-NEXT.md)
3. 查看详细进度: [.claude/PROGRESS-CURRENT.md](.claude/PROGRESS-CURRENT.md)

---

**推荐**: 使用uv代替conda/pip，享受极速的包管理体验！
