# Generated651 多阶段胃癌世界模型

本仓库整理 2026-09-26 完整重训版本，默认配置为 **G2 / bs4_baseline**。
该配置是 15 个配置各 20 个种子的实验中，按验证集平均 AP 选出的配置。
模型从 CT0 和临床、治疗、时间条件生成 S1，再经过手术条件转移生成 S2，分别预测 pCR 和记录的复发/转移状态。

## 网络和选择依据

- [网络结构](docs/ARCHITECTURE.md)：输入维度、S0/S1/S2、预测头和损失。
- [配置与历史结果](docs/SELECTION.md)：验证集选择规则及历史汇总。
- `configs/selected.json`：推荐配置快照；实际运行参数由 `regularization_spec.py` 定义。
- `reference/`：原始配置选择结果及按种子汇总的统计，不含逐患者记录。

## 安装与验证

```bash
python -m pip install -e '.[dev,binary-endpoints]'
python -m pytest -q tests/test_regularization.py tests/test_release_configuration.py
python scripts/run_regularization.py --help
```

Python 3.11+；正式训练使用支持 BF16 的 NVIDIA GPU。先安装与驱动匹配的 PyTorch。
缓存训练不需要下载 CT 基础编码器。验证记录见 [VALIDATION.md](docs/VALIDATION.md)。

## 数据与训练

需要授权的 complete651 特征缓存与对应的手术事件缓存。默认路径：

```text
data/complete651/artifacts/pool/pool.pt
data/surgery/artifacts/pool/events.pt
```

具体文件合同见 [DATA.md](docs/DATA.md)。也可在启动前设置
`GENERATED651_SOURCE_ROOT`、`GENERATED651_EVENT_ROOT` 指向各自含 `artifacts/pool/` 的目录。
缓存、权重和患者记录不随代码上传。

```bash
# 默认只运行推荐配置；smoke 仍需要真实的本地缓存。
python scripts/run_regularization.py --mode smoke --workers 1
python scripts/run_regularization.py --mode formal --workers 2

# 复现原始 15 配置实验矩阵。
python scripts/run_regularization.py --mode formal --arm all --workers 2
```

输出分别位于 `artifacts/bs4_baseline-formal/` 和 `artifacts/all-formal/`。
单配置重跑与原始 15 配置的选择实验分开记录；不要把单配置结果当成新的配置搜索。
中断恢复、训练内预处理、验证集选择、测试前锁定和独立推理导出保留原逻辑。

## 推理与源码

`stageworld.regularization_inference.predict_bundle` 接收临床字段、治疗条件、时间间隔、
CT0 的 `[B,27,768]` 特征和手术状态；不接收 CT1 或结局标签。
模型包位于 `fits/bs4_baseline/seed-*/inference.pt`。

主入口是 `regularization_workflow.py`；网络由 `regularization_models.py`、
`surgery_s2_models.py`、`generated700_models.py` 组成。`src/` 保留完整依赖模块。
本次仅整理代码、增加可迁移的路径和配置入口，没有重新进行完整临床训练。
历史数据为反复用于开发的内部队列，种子间测试患者有重叠，不是独立外部验证。

许可与第三方来源见 [NOTICE.md](NOTICE.md)。
