import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
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
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))  # 逻辑回归模型
])

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear', C=1, probability=True))  # 支持向量机模型
])

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))  # XGBoost模型
])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
lr_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)

# 预测结果
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_svm = svm_pipeline.predict(X_test)
y_pred_xgb = xgb_pipeline.predict(X_test)

# 计算评估指标
def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        'Accuracy': accuracy,
        'Precision (class 0)': report['0']['precision'],
        'Recall (class 0)': report['0']['recall'],
        'F1-score (class 0)': report['0']['f1-score'],
        'Precision (class 1)': report['1']['precision'],
        'Recall (class 1)': report['1']['recall'],
        'F1-score (class 1)': report['1']['f1-score']
    }

# 获取三个模型的评估结果
lr_metrics = get_metrics(y_test, y_pred_lr)
svm_metrics = get_metrics(y_test, y_pred_svm)
xgb_metrics = get_metrics(y_test, y_pred_xgb)

# 计算AUC值
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_pipeline.predict_proba(X_test)[:, 1])
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_pipeline.predict_proba(X_test)[:, 1])
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pipeline.predict_proba(X_test)[:, 1])

roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# 将结果汇总到一个数据框
metrics_df = pd.DataFrame([lr_metrics, svm_metrics, xgb_metrics], index=['Logistic Regression', 'SVM', 'XGBoost'])

# 添加AUC值到对比表
metrics_df['AUC'] = [roc_auc_lr, roc_auc_svm, roc_auc_xgb]

# 显示结果对比表
print(metrics_df)

# 绘制ROC曲线对比
plt.figure(figsize=(6, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'LR ROC curve (AUC = {roc_auc_lr:.2f})')
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label=f'SVM ROC curve (AUC = {roc_auc_svm:.2f})')
plt.plot(fpr_xgb, tpr_xgb, color='green', lw=2, label=f'XGBoost ROC curve (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
