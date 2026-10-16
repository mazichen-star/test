# 材料力学性能预测系统 (Streamlit)

基于 Streamlit 的材料力学性能回归预测工具，支持上传 CSV / Excel 数据，自动训练多种回归模型并输出预测结果与不确定性。

## 功能
- 上传 CSV / XLSX 数据，自动识别数值列与类别列（类别列自动 Label Encoding）
- 支持模型：线性回归、Ridge、Lasso、随机森林、梯度提升树、SVR、高斯过程回归(GPR)
- GPR 提供 3 种核函数（RBF+白噪声 / RBF / Matern 3/2），输出预测均值与 ±95% 置信区间
- 评估指标：R²、MAE、RMSE、5 折交叉验证 R²
- 特征重要性图、预测值 vs 真实值散点图、GPR 置信区间带
- 一键下载含预测列的 Excel 结果
- 手动输入新样本进行预测

## 安装
```bash
pip install -r requirements.txt
```

## 运行
```bash
streamlit run app.py
```

## 数据格式
每行为一个样本，列为工艺/成分参数，最后一列或指定列为目标力学性能（如抗拉强度 MPa、屈服强度、弹性模量、延伸率等）。

## 部署（Streamlit Community Cloud）
将仓库推到 GitHub 后，在 https://share.streamlit.io 选择该仓库，主文件填 `app.py` 即可免费部署。
