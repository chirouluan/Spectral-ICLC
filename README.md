# 光谱分类的大模型上下文学习基准

本项目使用冻结的大模型和传统分类器，评估四批红外/近红外光谱分类任务。模型使用带标签的参考样本及无标签查询特征；验证集和测试集标签用于预测后的离线评分。

## 数据与实验

`data_json/four_batches/` 当前选用 260 套实验包：同日分类 T1、跨日迁移 T2、未见浓度 UNSEEN 各 60 套，以及四个公共数据任务共 80 套。公共任务是咖啡品种二分类、橄榄油产地四分类、姜黄真伪二分类和牛肉掺假五分类。输入特征已经预处理或降维。

`index_current.csv` 对应当前选用的 260 套实验，是运行入口使用的索引。

## 安装与预览

需要 Python 3.10 或更高版本。在项目根目录运行：

```bash
python -m pip install -r requirements.txt
python run_variants.py --phase ALL --dry-run --limit 2
```

## 本地模型推理

启动 OpenAI 兼容的本地模型服务后，例如：

```bash
python run_variants.py --phase ALL --variant my_run --output-root outputs --no-thinking --format plain --max-tokens 16384 --concurrency 2
```
