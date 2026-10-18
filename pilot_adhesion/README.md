# PAA-DA 黏附预测（独立试用版）

这是主应用之外的**独立小试用版**，专门针对：

- 材料：聚丙烯酸-多巴胺（PAA-DA）类生物有机高分子
- 文献：约 3 篇 PDF
- 实验数据：约 30 条
- 目标：黏附力学性能（lap shear / peel / tack）

它不会修改主应用 `app.py`，也不会写 Supabase；所有数据都在 `pilot_adhesion/data/`。

## 启动

```powershell
cd "C:\Users\LENOVO\Videos\Users\LENOVO\Desktop\大创\app\test"
pip install -r pilot_adhesion/requirements.txt
streamlit run pilot_adhesion/app.py
```

如果只想用主应用，回退：

```powershell
git switch main
```

试用版文件会保留在 `pilot_adhesion/`，但主应用回到 main 的版本。

## 使用流程

### 1. 文献数据

1. 上传 3 篇 PDF；
2. 如果配置了 AI Key，点击“用 AI 抽取全部 PDF”；
3. 没有 Key 时，可以复制 JSON 模板，手动粘贴；
4. 检查抽取结果，确认单位、测试类型、页码和原文引用；
5. 通过后直接用于训练，或下载 CSV 备份。

AI 抽取字段：

- 多巴胺含量、聚合物浓度、pH
- 固化温度、固化时间
- 交联剂、基材
- 测试类型（lap_shear / peel / tack）
- 黏附强度、原始单位
- 测试温度、湿度
- 页码、原文引用、置信度

### 2. 实验数据

1. 上传 CSV / Excel（约 30 行）；
2. 选择目标列和原始单位；
3. 把中文列名映射到标准字段（多巴胺、浓度、pH、固化温度/时间、交联剂、基材、测试类型）；
4. 保存为标准 CSV。

也可以点“生成 30 条演示实验数据”先跑通流程。演示数据是合成的，**不能用于真实结论**。

### 3. 训练与预测

- 选择测试类型（不同方法不要混在一起）；
- 选择是否把文献数据加入训练集；
- 模型：GPR / 随机森林 / 梯度提升树 / Ridge；
- 30 条左右数据建议先试 GPR 或 Ridge；
- 输出 R²、MAE、RMSE、交叉验证 R²、预测 vs 真实；
- 预测新配方时，GPR 会给出 95% 预测区间；
- 页面会列出最相似的文献记录。

## AI 配置

复制：

```text
pilot_adhesion/.streamlit/secrets.toml.example
```

为：

```text
pilot_adhesion/.streamlit/secrets.toml
```

填写：

```toml
DEEPSEEK_API_KEY = "sk-..."
DEEPSEEK_MODEL = "deepseek-chat"
```

或使用火山方舟：

```toml
ARK_API_KEY = "..."
ARK_MODEL = "ep-..."
```

没有配置时，AI 抽取按钮不会工作，但手动 JSON 和预测模型仍可使用。

## 数据位置

```text
pilot_adhesion/data/pilot.db          # 文献记录
pilot_adhesion/data/experiments.csv   # 标准化实验数据
```

这两个文件默认不提交到 Git。要彻底清空，点侧边栏“清空试用数据”。

## 重要限制

- 3 篇文献 + 30 条实验数据是非常小的样本，模型只能作为趋势参考；
- 文献中的 lap shear、peel、tack 不能混在一起训练；
- 单位换算只覆盖常见单位；无法换算的记录会被忽略；
- AI 抽取必须人工检查，特别是页码、单位、测试条件和原文引用；
- 没有做 DOI 查重、图表数字化和扫描版 PDF OCR；
- 预测结果不能替代真实实验，尤其是新配方/新基材。

## 建议的下一步

1. 先用真实 3 篇 PDF + 30 条数据跑通“文献抽取 → 审核 → 训练”；
2. 检查文献记录的单位和测试类型是否一致；
3. 用同一测试类型分别建模；
4. 如果数据量增加，再加 SHAP、批量预测和主动学习。
