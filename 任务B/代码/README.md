# 核心代码

- `lrexp/`：本轮最终离线训练实现，包含任务生成器、Transformer、采样、训练及完整状态保存。
- `机制/`：报告使用的读取限制、表示探针、恢复与本地补充分析源码。

在仓库根目录安装依赖并查看训练入口：

```sh
python3 -m pip install -r 任务B/requirements.txt
cd 任务B/代码
python3 -m lrexp.train2 --help
```

训练命令为 `python3 -m lrexp.train2 --out runs/NAME --set key=value ...`。CUDA训练需使用支持CUDA的PyTorch。

实际246条1024与33条4096运行的模型、任务、训练参数均嵌入 [1024结果快照](../结果/1024/results_snapshot.json) 和 [4096结果快照](../结果/4096/results_snapshot.json) 的各行中，分别为 `model_config`、`task_config`、`train_config`。

源码保留执行时的算法，仅将机制核心模块的运行时定位调整为当前目录结构，并补齐权重加载所用的辅助模块。部分阶段分析脚本包含原工作路径，重算时需要提供本地权重与评测输入，并调整输入、输出路径。报告与结果文件可以直接在本仓库阅读。
