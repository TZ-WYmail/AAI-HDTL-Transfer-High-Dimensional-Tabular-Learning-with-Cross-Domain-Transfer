# AAI Project - 高维小样本二分类模型

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

这是一个高维小样本数据的二分类机器学习项目，使用8个不同的机器学习模型进行性能对比和分析。项目包含完整的数据预处理、模型训练、评估和可视化流程。

### 核心特点

- 📊 **高维数据处理**: 从12,700个特征降维至500个核心特征
- 🤖 **8模型对比**: Logistic Regression, SVM (Linear/RBF), Random Forest, Gradient Boosting, KNN, Naive Bayes, Neural Network
- 🎯 **最优ROC-AUC**: 0.9405 (Logistic Regression)
- 📈 **完整Pipeline**: 数据预处理 → 特征选择 → 模型训练 → 交叉验证 → 结果分析
- 📝 **详细文档**: 数据分析报告 + 模型性能分析报告

---

## 目录结构

```
AAI_Project/
├── data/                           # 数据目录
│   ├── train.csv                  # 训练数据 (196样本 × 12,701列)
│   ├── test_in_domain.csv         # 域内测试集 (84样本)
│   └── test_cross_domain.csv      # 跨域测试集 (200样本)
│
├── src/                           # 源代码
│   ├── __init__.py
│   ├── data_processing.ipynb      # 数据分析和可视化
│   └── model_training_and_evaluation.ipynb  # 模型训练和评估
│
├── docs/                          # 文档
│   ├── data_analysis_report.md           # 数据分析完整报告
│   ├── model_performance_analysis.md     # 模型性能分析报告
│   ├── data_processing_explanation.md    # 数据处理说明
│   └── images/                           # 可视化图表
│       ├── 01_label_distribution.png
│       ├── 02_feature_variance_distribution.png
│       ├── 03_feature_label_correlation.png
│       ├── 04_top_features_distribution.png
│       ├── 05_top_features_boxplot.png
│       ├── 06_pca_analysis.png
│       └── 07_model_comparison.png
│
├── requirements.txt               # Python依赖
├── .gitignore                     # Git忽略文件
└── README.md                      # 项目说明文档
```

---

## 快速开始

### 1. 环境要求

- Python 3.12+
- Jupyter Notebook / JupyterLab
- 推荐使用虚拟环境

### 2. 安装依赖

```bash
# 创建虚拟环境（可选）
python -m venv venv

# Windows激活虚拟环境
venv\Scripts\activate

# Linux/Mac激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行数据分析

```bash
# 启动Jupyter Notebook
jupyter notebook

# 打开并运行
src/data_processing.ipynb
```

### 4. 训练和评估模型

```bash
# 运行模型训练notebook
src/model_training_and_evaluation.ipynb
```

---

## 数据集信息

### 训练集 (train.csv)

| 属性 | 值 |
|------|-----|
| 样本数 | 196 |
| 特征数 | 12,700 |
| 标签列 | 1 (最后一列) |
| 类别分布 | 类别0: 70 (35.7%), 类别1: 126 (64.3%) |
| 类别比例 | 1:1.8 (轻微不平衡) |
| 缺失值 | 0 |
| 常量特征 | 0 |

### 测试集

- **test_in_domain.csv**: 84个样本（域内测试）
- **test_cross_domain.csv**: 200个样本（跨域测试）

---

## 核心功能

### 1. 数据预处理 Pipeline

```
原始数据 (196 × 12,700)
    ↓
方差过滤 (threshold=0.01) → 移除531个低方差特征
    ↓
标准化 (StandardScaler) → Z-score归一化
    ↓
特征选择 (SelectKBest, k=500) → 选择Top 500特征
    ↓
最终数据 (196 × 500)
```

**降维比例**: 96.1%

### 2. 模型训练与评估

#### 8个机器学习模型

| 模型 | ROC-AUC | Accuracy | F1-Score | 训练时间 | 推荐度 |
|------|---------|----------|----------|----------|--------|
| **Logistic Regression** | **0.9405** | **0.8676** | 0.8940 | 0.004s | ⭐⭐⭐⭐⭐ |
| **SVM (Linear)** | **0.9360** | **0.8676** | **0.8960** | 0.009s | ⭐⭐⭐⭐⭐ |
| **SVM (RBF)** | **0.9078** | 0.8065 | 0.8465 | 0.010s | ⭐⭐⭐⭐ |
| Neural Network (MLP) | 0.8645 | 0.7915 | 0.8342 | 1.107s | ⭐⭐⭐⭐ |
| Naive Bayes | 0.8424 | 0.7303 | 0.7753 | 0.001s | ⭐⭐⭐ |
| Random Forest | 0.8407 | 0.7656 | 0.8355 | 0.258s | ⭐⭐⭐ |
| Gradient Boosting | 0.8131 | 0.7603 | 0.8296 | 5.336s | ⭐⭐⭐ |
| K-Nearest Neighbors | 0.8018 | 0.7097 | 0.8059 | 0.001s | ⭐⭐ |

#### 评估指标

- **主指标**: ROC-AUC (受类别不平衡影响小)
- **辅助指标**: Accuracy, F1-Score, Precision, Recall
- **验证方法**: 5折分层交叉验证 (StratifiedKFold)

### 3. 最优模型配置

**Logistic Regression** (ROC-AUC: 0.9405) 🏆

```python
LogisticRegression(
    penalty='l2',              # L2正则化
    C=0.1,                    # 强正则化防止过拟合
    max_iter=2000,            # 最大迭代次数
    class_weight='balanced',   # 自动处理类别不平衡
    random_state=42
)
```

**为什么Logistic Regression最优？**
- ✅ 最高ROC-AUC (0.9405)
- ✅ 最稳定 (Std=0.0179)
- ✅ 训练快速 (0.004秒)
- ✅ 高可解释性

---

## 主要发现

### 数据特征

- ✅ **数据质量良好**: 无缺失值、无常量特征
- ⚠️ **高维度挑战**: 12,700个特征，远超样本数
- ⚠️ **小样本限制**: 仅196个训练样本
- ⚠️ **低信噪比**: 71%的特征与标签相关性<0.1
- ✅ **轻微类别不平衡**: 1:1.8比例，可处理

### 模型性能

1. **线性模型显著优于非线性模型**
   - 前3名均为线性或核方法
   - Logistic Regression和SVM (Linear)性能接近

2. **简单往往更好** (Occam's Razor)
   - 最简单的Logistic Regression击败所有复杂模型
   - 正则化比模型复杂度更重要

3. **过拟合风险**
   - 训练集与验证集存在轻微差距
   - 强正则化(C=0.1)是关键

---

## 使用示例

### 完整训练流程

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. 加载数据
train_df = pd.read_csv('data/train.csv')
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

# 2. 构建Pipeline
pipeline = Pipeline([
    ('variance_filter', VarianceThreshold(threshold=0.01)),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=500)),
    ('classifier', LogisticRegression(C=0.1, max_iter=2000, 
                                     class_weight='balanced', 
                                     random_state=42))
])

# 3. 交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(pipeline, X, y, cv=cv, 
                           scoring='roc_auc', return_train_score=True)

print(f"ROC-AUC: {cv_results['test_score'].mean():.4f} ± {cv_results['test_score'].std():.4f}")

# 4. 训练最终模型
pipeline.fit(X, y)

# 5. 预测测试集
test_df = pd.read_csv('data/test_in_domain.csv')
predictions = pipeline.predict_proba(test_df.values)[:, 1]

# 6. 保存结果
pd.DataFrame({'prediction_proba': predictions}).to_csv('predictions_in_domain.csv', index=False)
```

### 多模型比较

```python
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

models = {
    'Logistic Regression': LogisticRegression(C=0.1, max_iter=2000, random_state=42),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
}

for name, model in models.items():
    pipeline = Pipeline([
        ('variance_filter', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=500)),
        ('classifier', model)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 文档

### 详细分析报告

1. **[数据分析报告](docs/data_analysis_report.md)** (1,500+ 行)
   - 数据集基本信息
   - 特征分布和相关性分析
   - PCA降维分析
   - 建模策略建议
   - 完整Pipeline示例

2. **[模型性能分析报告](docs/model_performance_analysis.md)** (500+ 行)
   - 8个模型详细对比
   - 性能指标多维度分析
   - 最优模型深度剖析
   - 场景化模型选择建议
   - 性能优化路线图

### 可视化图表

- 标签分布图
- 特征方差分布图
- 特征-标签相关性热图
- Top特征分布对比
- PCA降维分析图
- 模型性能对比图（4维度）

---

## 性能优化建议

### 短期优化 (快速见效)

1. **超参数精细调优**
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {
       'C': [0.01, 0.05, 0.1, 0.5, 1.0],
       'penalty': ['l1', 'l2']
   }
   
   grid_search = GridSearchCV(LogisticRegression(max_iter=2000), 
                              param_grid, cv=5, scoring='roc_auc')
   ```
   **预期提升**: +0.005-0.01 ROC-AUC

2. **特征数量优化**
   - 测试k值: 200, 300, 400, 500, 600, 800
   - 当前k=500可能不是最优

3. **集成学习**
   ```python
   from sklearn.ensemble import VotingClassifier
   
   ensemble = VotingClassifier([
       ('lr', LogisticRegression(C=0.1)),
       ('svm', SVC(kernel='linear', probability=True))
   ], voting='soft')
   ```
   **预期提升**: +0.01-0.02 ROC-AUC

### 中期优化

- SMOTE过采样处理类别不平衡
- 特征交互项生成
- 深度学习架构优化

### 长期优化

- 重新设计特征选择策略 (RFECV)
- 探索其他降维方法 (PCA, UMAP)
- 高级集成策略 (Stacking)

---

## 常见问题

### Q1: 为什么不使用深度学习？

**A**: 
- 样本量太小（196个），深度学习容易过拟合
- 神经网络(MLP)表现中等（ROC-AUC 0.8645），不如简单的Logistic Regression
- 训练时间长（1.1秒 vs 0.004秒）

### Q2: 如何处理类别不平衡？

**A**: 
- 使用`class_weight='balanced'`自动调整权重
- 评估指标选择ROC-AUC（对不平衡不敏感）
- 可选：SMOTE过采样（需要注意数据泄露）

### Q3: 为什么降维这么重要？

**A**: 
- 特征数(12,700) >> 样本数(196)，维度诅咒严重
- 不降维会导致严重过拟合
- 降维至500维后，模型性能显著提升

### Q4: 如何避免数据泄露？

**A**: 
- 使用sklearn的Pipeline
- 特征选择在交叉验证的每一折中独立进行
- 测试集不参与任何训练过程

---

## 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 如何贡献

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

- **项目作者**: AAI Team
- **创建日期**: 2025年12月7日
- **最后更新**: 2025年12月7日

---

## 致谢

- scikit-learn 团队提供的优秀机器学习库
- matplotlib 和 seaborn 提供的可视化工具
- Jupyter 项目提供的交互式开发环境

---

## 更新日志

### v1.0.0 (2025-12-07)
- ✨ 初始版本发布
- ✅ 完成数据分析和可视化
- ✅ 训练和评估8个机器学习模型
- ✅ 生成详细的分析报告
- ✅ 确定最优模型 (Logistic Regression, ROC-AUC: 0.9405)

---

**⭐ 如果这个项目对你有帮助，请给一个Star！**
