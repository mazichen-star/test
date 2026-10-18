# 材料力学性能预测系统

一个基于 Streamlit 的 Web 应用，面向材料研发、工艺优化和实验数据分析。用户不需要写机器学习代码，上传记录了“成分 / 工艺参数 → 力学性能”的 CSV / Excel 后，即可完成数据清洗、模型训练、对比评估、可视化、导出和新样本预测。

典型目标：抗拉强度、屈服强度、弹性模量、延伸率、硬度等。

## 1. 主要功能

- **数据读取**：CSV（UTF-8 / GBK / GB18030 自动回退）、XLSX、XLS；Excel 实验报告可自动识别表头、跳过标题行/单位行/尾部签名行。
- **可选 AI 清洗**：配置大模型 Key 后，把表格前 50 行交给模型识别表头和数据区；不配置也能正常使用。
- **数据预处理**：数值转换、类别标签编码、缺失行剔除；目标列必须是数值。
- **十种回归模型**：线性回归、Ridge、Lasso、随机森林、梯度提升树、极端随机树、SVR、核岭回归、MLP、高斯过程回归（GPR）。
- **模型对比**：同时训练多个模型，按测试集 R² 排名并自动选最优；失败模型不会拖垮整个流程。
- **指标与解释**：R²、MAE、RMSE、交叉验证 R²、特征重要性、预测 vs 真实、残差、特征响应曲线。
- **GPR 不确定性**：输出 95% 预测区间。
- **导出**：带预测列的 Excel、Word 训练报告、贝叶斯寻优结果 CSV。
- **新样本预测**：留空数值用训练均值、未知类别用众数填充；预测结果在页面重跑后仍保留。
- **贝叶斯优化**：基于训练模型搜索数值特征的最优组合，使用真实训练 min/max 作为搜索边界；未安装 `scikit-optimize` 时自动禁用。
- **共享材料库**：按化学式搜索、录入、批量导入、文献正则提取、材料性能对比；删除只能删自己录入的记录。
- **账号系统**：注册、登录、改密、找回密码、删除账号；密码使用 PBKDF2-HMAC-SHA256 + 随机盐，兼容旧版 SHA-256 哈希并自动升级。
- **存储**：有 Supabase 配置时用云数据库；没有配置时自动使用本地 SQLite，开箱即用。
- **界面**：中英文切换、暗色模式、窄屏适配。

## 2. 本次完善的重点

### 修复的确定性 bug

- 修复批量导入缩进错误：之前 `rows_to_insert.append(...)` 在循环外，未点击按钮时还会 `NameError`；现在逐行导入、去重、错误提示完整。
- 修复“文献NLP提取”标签页挂到材料对比区的问题，NLP 界面回到材料库内。
- 修复 Word 报告对 NumPy 数组使用 `.iloc` 导致的 `AttributeError`。
- 修复训练结果只在按钮点击那一次显示的问题：结果写入 `session_state`，切换页面/改语言/下载后不会消失。
- 修复上传文件在每次 rerun 时重复入库的问题：按内容哈希只保存一次，并按 `(username, filename)` 更新。
- 修复 GPR 区间文案：`±1.96σ` 是预测区间，不是置信区间。
- 修复贝叶斯优化边界使用 `mean ± 2σ`、收敛曲线按排序后数据绘制的问题；现在使用真实 min/max 和原始迭代顺序。
- 修复材料对比把 MPa/GPa/S/m 画在同一根柱状图上的问题，改为每个指标独立子图。
- 修复 CSV 编码、`.xls` 依赖、`UploadedFile` 缓存、AI 清洗缓存签名等数据读取问题。

### 安全与稳定性

- 找回密码**不再在页面上显示临时密码**；邮件发送失败会回滚密码。未配置 SMTP 时禁用自助找回。
- 密码从无盐 SHA-256 升级为 PBKDF2；旧账号登录时自动升级。
- 注册、改密、删除账号都检查数据库返回，不再“失败也提示成功”。
- 退出登录/删除账号会清除模型、数据和页面状态，避免下一个账号看到上一个账号的模型。
- 交叉验证改为 `Pipeline(StandardScaler, model)`，避免先缩放造成数据泄漏；小样本自动减少折数或跳过。
- 模型训练/预测/寻优都加了异常处理，错误以页面提示呈现。

## 3. 项目结构

```text
.
├── app.py                          # Streamlit 页面与交互
├── core/
│   ├── config.py                   # 环境变量 / Streamlit Secrets
│   ├── security.py                 # PBKDF2 密码哈希与校验
│   ├── storage.py                  # Supabase + SQLite 两套存储
│   ├── data.py                     # 读取、表头识别、预处理、签名
│   ├── modeling.py                 # 模型注册、训练、评估、CV
│   └── reporting.py                # Excel / Word 导出
├── tests/
│   └── test_core.py                # 核心逻辑测试
├── fonts/
│   └── NotoSansSC-Regular.ttf      # 图表中文字体
├── schema.sql                      # Supabase 建表 SQL
├── requirements.txt
├── requirements-optional.txt       # scikit-optimize（贝叶斯优化）
├── requirements-dev.txt            # pytest
├── .streamlit/
│   └── secrets.toml.example
├── README.md
└── LICENSE
```

## 4. 快速开始（本地 SQLite，零配置）

要求 Python 3.10+。

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

启动后打开浏览器，注册账号即可使用。默认数据保存在 `data/app.db`，该目录已被 `.gitignore` 忽略。

需要贝叶斯优化时再安装：

```bash
pip install -r requirements-optional.txt
```

## 5. 使用 Supabase（可选）

1. 在 Supabase 创建项目。
2. 打开 SQL Editor，执行 `schema.sql`。
3. 复制配置模板：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

4. 填写：

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

5. 重启应用。页面侧边栏会显示“存储：Supabase 云数据库”。

没有填写 Supabase 配置时，应用不会报错退出，而是自动使用本地 SQLite。

> 安全提示：当前架构是“自定义登录 + 请求里携带 username”，Supabase RLS 无法据此区分终端用户。公网部署建议改用 Supabase Auth，或把数据库访问放到自己的后端 API 中；不要在前端/Streamlit 中使用 `service_role` key。

## 6. 邮件找回密码（可选）

在 `.streamlit/secrets.toml` 中配置：

```toml
SMTP_HOST = "smtp.example.com"
SMTP_PORT = 465
SMTP_USER = "noreply@example.com"
SMTP_PASSWORD = "your-smtp-password"
SMTP_USE_SSL = true
```

- 465 端口通常用 `SMTP_USE_SSL = true`。
- 587 端口通常用 `SMTP_USE_SSL = false`（STARTTLS）。
- 未配置 SMTP 时，“忘记密码”会禁用；页面不会输出临时密码。
- 发送失败会回滚密码，并提示稍后重试或联系管理员。

## 7. AI 表格清洗（可选）

配置以下任意一组：

```toml
DEEPSEEK_API_KEY = "sk-..."
DEEPSEEK_MODEL = "deepseek-chat"

# 或者
ARK_API_KEY = "..."
ARK_MODEL = "ep-..."
```

启用后，页面会显示“AI 智能清洗表格结构”开关。只会发送表格前 50 行，不会上传整个文件；关闭开关即使用本地规则清洗。

## 8. 数据格式建议

- 第一行是列名，每行一个样本。
- 至少包含一列数值型目标（如 `Tensile`、`抗拉强度`）。
- 特征列可以是数值或文本类别（如 `Structure=FCC/BCC`）。
- 目标列不能有无法转成数字的值；这些行会被剔除。
- 缺失特征行会被剔除，页面会显示剔除数量。
- 列名建议避免重复；重复列名会自动加后缀。

示例：

```csv
C,Mn,Temp,Structure,Tensile
0.2,1.1,900,FCC,450
0.4,1.3,950,BCC,510
```

## 9. 测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

测试不依赖 Streamlit 或 Supabase，覆盖密码哈希、SQLite 存储、预处理、Excel 表头识别、模型训练和 Excel/Word 导出。

## 10. 部署

### Streamlit Community Cloud

1. 把仓库推到 GitHub。
2. 在 Streamlit Cloud 新建应用，入口选择 `app.py`。
3. 在 Secrets 中粘贴与 `.streamlit/secrets.toml` 相同的配置。

### 其他平台（SealOS / Docker / 服务器）

- 启动命令：`streamlit run app.py --server.address 0.0.0.0 --server.port 8501`
- 通过环境变量注入 `SUPABASE_URL`、`SUPABASE_KEY`、`SMTP_*`、`DEEPSEEK_API_KEY` 等。
- 如果使用本地 SQLite，确保 `data/` 目录可写并做持久化挂载。

## 11. 已知限制

- 这是回归工具，不处理分类目标。
- 特征响应曲线固定其他特征为均值/众数，只反映模型局部行为，不能当作因果效应。
- 模型选择默认按测试集 R² 排序，存在一定乐观偏差；严谨实验建议使用独立验证集或嵌套交叉验证。
- 还没有模型文件保存/加载、批量新样本预测、SHAP 解释和超参数搜索。
- 公网部署前请补充登录限流、邮箱验证和更严格的 Supabase 权限方案。

## 12. License

Apache License 2.0，见 `LICENSE`。
