# 材料力学性能预测系统

基于 Streamlit 的材料力学性能回归预测 Web 应用，支持上传 CSV / Excel 数据，自动训练多种机器学习模型，预测材料的力学性能（抗拉强度、屈服强度、弹性模量、延伸率等）。

## 功能特性

- **多模型支持**：线性回归、Ridge、Lasso、随机森林、梯度提升树、极端随机树、SVR、核岭回归、MLP 神经网络、高斯过程回归(GPR)
- **多模型对比**：一次训练多个模型，自动按 R² 排名并推荐最优模型
- **可视化分析**：
  - 预测值 vs 真实值散点图
  - 特征重要性条形图
  - 单特征拟合曲线（标注预测极值点）
  - GPR 95% 置信区间带
- **用户系统**：注册 / 登录 / 找回密码（SMTP 邮件发送临时密码）
- **数据管理**：上传文件云端持久化，按用户隔离，支持历史文件加载
- **自适应界面**：响应式布局，电脑 / 手机均可使用；暗色模式；中英文切换
- **结果导出**：一键下载含预测列的 Excel

## 技术栈

- **前端**：Streamlit
- **机器学习**：scikit-learn
- **数据处理**：pandas / numpy
- **可视化**：matplotlib
- **数据库**：Supabase (PostgreSQL)
- **邮件服务**：SMTP（QQ / 163 / Gmail 等）

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/mazichen-star/test.git
cd test

# 安装依赖
pip install -r requirements.txt

# 启动
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 在线部署

本应用已部署在 Streamlit Community Cloud：

1. Fork 本仓库
2. 打开 [share.streamlit.io](https://share.streamlit.io)
3. 选择仓库，主文件填 `app.py`
4. 在 **Settings → Secrets** 中配置以下密钥：

```toml
# Supabase 云数据库（在 supabase.com 建项目后获取）
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_KEY = "你的 anon key"

# SMTP 邮件服务（用于找回密码）
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = "465"
SMTP_USER = "your@qq.com"
SMTP_PASSWORD = "你的SMTP授权码"
```

## 数据库初始化

首次使用需在 Supabase SQL Editor 执行：

```sql
CREATE TABLE users (
  username TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,
  email TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_files (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  username TEXT NOT NULL,
  filename TEXT NOT NULL,
  content BYTEA NOT NULL,
  uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE users DISABLE ROW LEVEL SECURITY;
ALTER TABLE user_files DISABLE ROW LEVEL SECURITY;

GRANT SELECT, INSERT, UPDATE, DELETE ON public.users TO anon;
GRANT SELECT, INSERT, DELETE ON public.user_files TO anon;
GRANT USAGE, SELECT ON SEQUENCE public.user_files_id_seq TO anon;
```

## 数据格式

每行为一个样本，列为工艺/成分参数与目标力学性能，例如：

| 碳含量 | 温度 | 应变速率 | 抗拉强度(MPa) |
|--------|------|----------|---------------|
| 0.22   | 850  | 0.01     | 456           |
| 0.18   | 900  | 0.1      | 421           |

将目标列（如抗拉强度）设为预测目标，其余列设为特征即可。

## 项目结构

```
├── app.py              # 主程序
├── requirements.txt    # Python 依赖
├── README.md
├── .gitignore
└── fonts/
    └── NotoSansSC-Regular.ttf   # 中文字体（解决 matplotlib 中文方块）
```

## 使用说明

1. 注册账号并登录
2. 上传数据文件（CSV / XLSX），文件自动保存到云端
3. 选择目标列（要预测的力学性能）和特征列（输入参数）
4. 选择模型，或开启多模型对比
5. 调整训练轮数、测试集比例，点击"开始训练与预测"
6. 查看评估指标（R² / MAE / RMSE / 交叉验证）和图表
7. 在侧边栏用训练好的模型预测新样本，或下载含预测列的 Excel
