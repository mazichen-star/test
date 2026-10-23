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
- **多文件合并**：一次上传多个同类型 CSV / Excel，按列名合并成一个数据集；列不一致时按并集合并并提示差异，可下载合并后的 CSV。
- **新样本预测**：留空数值用训练均值、未知类别用众数填充；预测结果在页面重跑后仍保留。
- **贝叶斯优化**：基于训练模型搜索数值特征的最优组合，使用真实训练 min/max 作为搜索边界；未安装 `scikit-optimize` 时自动禁用。
- **共享材料库**：按化学式搜索、录入、批量导入、文献正则提取、材料性能对比；删除只能删自己录入的记录。
- **PAA-DA 黏附模块**：上传 PDF，由 AI 抽取聚丙烯酸-多巴胺的黏附数据；与实验数据一起训练 GPR/随机森林/梯度提升/Ridge，输出预测区间和相似文献；数据保存到 Supabase。
- **账号系统**：Supabase 模式使用 **Supabase Auth + RLS**（邮箱登录、邮箱找回、JWT 绑定用户）；本地 SQLite 调试模式使用 PBKDF2-HMAC-SHA256 + 随机盐，兼容旧版 SHA-256 哈希。
- **存储**：线上使用 Supabase；本地未配置 Supabase 时自动使用 SQLite 仅作调试，Streamlit Cloud 上请务必配置 Supabase。
- **界面**：统一浅色实验室工作台主题、中英文切换、窄屏适配；表格与指标使用同一套字阶和留白。
- **图表文字**：英文/数字使用 Times New Roman，中文使用宋体；每张图可在页面内编辑中英文标题和横纵轴标题。

## 2. 本次完善的重点

### 修复的确定性 bug

- 修复批量导入缩进错误：之前 `rows_to_insert.append(...)` 在循环外，未点击按钮时还会 `NameError`；现在逐行导入、去重、错误提示完整。
- 修复“文献提取”标签页挂到材料对比区的问题，提取界面回到材料库内。
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
- Supabase 模式从“自建 users 表 + anon key”升级为 **Supabase Auth + RLS**：请求携带用户 JWT，RLS 用 `auth.uid()` 和 `profiles.username` 限制到本人；旧 `users` 表已锁死。

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
│   ├── adhesion.py                 # PAA-DA 文献抽取、单位换算、小样本模型
│   ├── supabase_auth.py            # Supabase Auth 登录、刷新、找回、删号
│   ├── plotting.py                 # 图表字体：Times 数字 + 宋体中文
│   └── reporting.py                # Excel / Word 导出
├── tests/
│   ├── test_core.py                # 核心逻辑测试
│   ├── test_adhesion.py            # 黏附模块测试
│   └── test_supabase_auth.py       # Auth / JWT 请求头测试
├── fonts/
│   ├── NotoSansSC-Regular.ttf      # 界面/备用中文字体
│   └── NotoSerifSC-VF.ttf          # 图表宋体（英文/数字用 Times）
├── schema.sql                      # Supabase 建表 SQL
├── requirements.txt
├── requirements-optional.txt       # scikit-optimize（贝叶斯优化）
├── requirements-dev.txt            # pytest
├── .streamlit/
│   ├── config.toml                 # 主题色与上传大小
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
2. **Authentication → Providers** 启用 Email；按需关闭 “Confirm email” 方便首次试用。
3. 打开 SQL Editor，执行 `schema.sql`。它会创建 `profiles`、Auth trigger、RLS 策略和 `delete_my_account()`。
5. 复制配置模板：

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

6. 填写：

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"
```

7. 重启应用。页面侧边栏会显示“存储：Supabase 云数据库”，登录页会变成邮箱登录。

没有填写 Supabase 配置时，应用不会报错退出，而是自动使用本地 SQLite。

`PAA-DA 黏附` 模块需要额外两张表，已经包含在 `schema.sql` 中：

- `adhesion_literature`：AI 从 PDF 抽取的文献数据；
- `adhesion_experiments`：标准化后的 30 条实验数据。

如果 Supabase 里已经有旧表，重新执行一次 `schema.sql` 即可（`create table if not exists` 不会删除已有数据）。

> 安全提示：Supabase 模式现在使用 Supabase Auth + RLS。`anon` 角色不再拥有业务表读写权限；所有查询都带用户 JWT，RLS 通过 `auth.uid()` 和 `profiles.username` 限制到本人。不要把 `service_role` key 放进 Streamlit 或浏览器。

### 5.0 旧数据迁移（可跳过）

如果你不需要旧账号和旧数据，直接跳过本节：执行 `schema.sql` 后，打开应用用邮箱注册一个新账号即可。

旧的 `public.users` 表（用户名 + PBKDF2 密码）在云模式下不再用于登录，并且已对 `anon/authenticated` 锁死。若要保留旧业务数据，按下面迁移：

1. Supabase Dashboard → **Authentication → Users → Add user**；
2. 邮箱填旧账号的邮箱，密码设置一个新密码；
3. User Metadata 填：

```json
{"username": "旧用户名"}
```

4. 保存后 trigger 会创建 `profiles`；
5. 旧 `user_files`、`adhesion_literature`、`adhesion_experiments`、`material_library` 按 `username` 自动对上，无需搬数据。

如果旧数据不需要保留，也可以让用户直接用同一用户名重新注册。

### 5.1 PAA-DA 黏附模块

正式版新增顶级标签「PAA-DA 黏附」，分三个子页：

1. **文献数据**：上传 PDF，AI 抽取 PAA-DA 黏附数据（多巴胺含量、浓度、pH、固化条件、基材、测试类型、黏附强度等），人工核对后保存到 `adhesion_literature`。
2. **实验数据**：上传 CSV / Excel，映射目标列和标准字段，保存到 `adhesion_experiments`。
3. **训练与预测**：文献 + 实验联合训练，GPR 输出 95% 预测区间，并列出最相似文献；配置 AI Key 后可生成带引用的解释。

AI 配置与第 7 节相同（`DEEPSEEK_API_KEY` / `ARK_API_KEY`）。没有 AI Key 时仍可手动粘贴 JSON，并使用模型训练与预测。

> 该模块的数据以 Supabase 为准；本地 SQLite 只在未配置 Supabase 时用于调试，Streamlit Cloud 重启后不会保留。

## 6. 找回密码

### Supabase 模式

找回密码走 **Supabase Auth**：

- 在 Supabase Dashboard → Authentication → Email 配置 SMTP；
- 应用调用 `/auth/v1/recover`，由 Supabase 发送重置密码邮件；
- 不需要在 Streamlit 里配置 `SMTP_*`，也不会生成临时密码。

### 本地 SQLite 调试模式

本地模式才使用自定义 SMTP：

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

测试不依赖 Streamlit 或 Supabase，覆盖密码哈希、SQLite 存储、预处理、Excel 表头识别、模型训练、Excel/Word 导出，以及 PAA-DA 的单位换算、实验标准化和小样本训练。

## 10. 部署

### Streamlit Community Cloud

1. 把仓库推到 GitHub 的 `main` 分支。
2. 在 Streamlit Cloud 新建应用，入口选择 `app.py`。
3. 在 Secrets 中至少配置：

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key"

# AI 文献抽取 / 解释（可选，但 PAA-DA 模块需要）
DEEPSEEK_API_KEY = "sk-..."
DEEPSEEK_MODEL = "deepseek-chat"
```

4. 在 Supabase SQL Editor 执行最新的 `schema.sql`，确保 `profiles`、`adhesion_literature`、`adhesion_experiments` 表存在，且 RLS 已启用。
5. 在 Supabase Authentication 中创建/迁移用户；旧业务数据按 `profiles.username` 关联。
6. 推送 `main` 后，Streamlit Cloud 会自动重新部署；如果没更新，在控制台点 “Reboot app”。

### 其他平台（SealOS / Docker / 服务器）

- 启动命令：`streamlit run app.py --server.address 0.0.0.0 --server.port 8501`
- 通过环境变量注入 `SUPABASE_URL`、`SUPABASE_KEY`、`SMTP_*`、`DEEPSEEK_API_KEY` 等。
- PAA-DA 模块以 Supabase 为准；不要依赖容器本地 SQLite 持久化。

## 11. 已知限制

- 这是回归工具，不处理分类目标。
- 特征响应曲线固定其他特征为均值/众数，只反映模型局部行为，不能当作因果效应。
- 模型选择默认按测试集 R² 排序，存在一定乐观偏差；严谨实验建议使用独立验证集或嵌套交叉验证。
- 还没有模型文件保存/加载、批量新样本预测、SHAP 解释和超参数搜索。
- Supabase Auth 已接入 RLS，但还没有登录限流、异常登录监测和审计日志；公网高流量场景建议再加一层网关/限流。

## 12. License

Apache License 2.0，见 `LICENSE`。
