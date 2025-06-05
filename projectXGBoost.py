import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBClassifier  # 导入XGBoost分类器
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 加载数据
data = pd.read_csv('D:/luo/train.csv')

# 删除不相关的列
data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# 删除 'Cabin' 列，因为它有太多缺失值
data.drop('Cabin', axis=1, inplace=True)

# 特征和目标变量
X = data.drop('Survived', axis=1)
y = data['Survived']

# 预处理管道：
# 1. 对数值特征使用 StandardScaler 标准化
# 2. 对类别特征使用 OneHotEncoder 进行独热编码
# 3. 使用 SimpleImputer 填充缺失值
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # 使用中位数填充缺失值
            ('scaler', StandardScaler())  # 标准化数值特征
        ]), ['Age', 'Fare']),  # 需要标准化的列

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # 用最频繁的值填充缺失值
            ('onehot', OneHotEncoder(handle_unknown='ignore'))  # 使用独热编码
        ]), ['Sex', 'Embarked'])  # 需要独热编码的列
    ])

# 创建一个包含预处理和分类器的管道
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))  # 使用XGBoost分类器
])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
pipeline.fit(X_train, y_train)

# 预测
y_pred = pipeline.predict(X_test)

# 评估模型
print(f"准确率: {accuracy_score(y_test, y_pred)}")
print("混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
print("分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 计算ROC曲线的假阳性率和真正率
fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# --- 学习曲线可视化 ---
train_sizes, train_scores, test_scores = learning_curve(
    pipeline, X, y, cv=5, n_jobs=-1, train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
)

# 计算平均和标准差
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# 绘制学习曲线
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, color='blue', label='Training Score')
plt.plot(train_sizes, test_mean, color='green', label='Cross-validation Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.1)
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

