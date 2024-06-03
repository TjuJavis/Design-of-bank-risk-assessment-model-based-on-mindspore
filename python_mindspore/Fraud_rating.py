
######################################################################################################
# 请点击左下方打开终端（alt+F12），输入conda activate mindspore进入环境，再输入python Fraud_rating.py运行程序
######################################################################################################

#### 导入相关库
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

#### Step1：导入相关库和实验数据
con_his = pd.read_csv("file_list/Consumption_history.csv", index_col=0) #读取数据。index_col=0：读取时不自动添加行号。


#### 数据预处理
### Step2：缺失值处理
print('判断是否含有缺失值中...')
print(con_his.isnull().sum())
print('未发现缺失值，处理完毕！')


### Step3：重复值处理
print('正在对重复值进行处理，即将删除多余的重复值，并更新索引...')
con_his.drop_duplicates(inplace=True)
con_his.reset_index(drop=True, inplace=True)
print('重复值处理完成！')


### Step4：异常值处理
# 查看数据基本特征
print('正在查看数据基本特征...')
print(con_his.describe())

## 为便于后续的可视化及数据分析工作，对一些数据中与模型建立求解准确度显然无关的变量"卡类别", "币种代码", "额度"删去，
## 并将各列用英文表示
#  日均消费金额	日均次数	单笔消费最小金额	单笔消费最大金额	个人收入_连续	是否存在欺诈
con_his = con_his.drop(columns=["卡号", "卡类别", "币种代码", "额度"])
con_his.columns = range(len(con_his.columns))
con_his.columns = ['daily_consumption', 'daily_number', 'minimum', 'maximum', 'personal_income', 'fraud_if']
print('数据中包含的特征（英文表示）为：')
print(con_his.columns)

## 分析发现，req_cus中可能存在的异常值有两方面，即年龄和工作年限的大小关系、个人年收入的大小。故对其进行处理。
unreal0 = 0
for index, row in con_his.iterrows():
    if row['maximum'] < row['minimum']:
        print("发现异常值！最大消费小于最小消费：", row['maximum'], "<", row['minimum'])
        temp = row['maximum']
        con_his.at[index, 'maximum'] = row['minimum']
        con_his.at[index, 'minimum'] = temp
        unreal0 = unreal0 + 1

print('已将异常值全部处理完毕！共计出现：', unreal0, '处异常值。')

# 绘制每个特征的分布，自定义数据框数据集全部特征分布图
def plot_distribution(df, cols=5, width=15, height=15, hspace=0.2, wspace=0.5):
    import math
    plt.style.use('seaborn-whitegrid')  # 设置绘画图表风格
    fig = plt.figure(figsize=(width, height))  # 创建figure实例
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)  # 调整图表位置和大小间距
    rows = math.ceil(float(df.shape[1]) / cols)  # ceil方法向上取整
    for i, column in enumerate(df.columns):  # 返回索引和其对应的列名
        ax = fig.add_subplot(rows, cols, i + 1)  # 创建子图，类似于subplot方法，返回的ax是坐标轴实际画图的位置，参数（子图总行数，总列数，子图位置）
        ax.set_title(column)  # 设置轴的标题
        if df.dtypes[column] == np.object:  # 通过列的类型来区分所选取的图像类型,
            g = sns.countplot(y=column, data=df)  # 属性类型为np.object时，countplot使用条形显示每个分箱器中的观察计数，y轴上的条形图
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(df[column])  # 不属于np.object类型即绘制 密度分布图
            plt.xticks(rotation=25)

print("正在绘制每个特征的分布...")
plot_distribution(con_his, cols=5, width=30, height=30, hspace=0.55, wspace=0.5)
plt.show()
print("绘制完成!")


### Step5：数据编码
print('正在进行数据编码...')
print('数据均为数字，未发现需编码数据！')
print(con_his)


### Step6：相关性分析
# 数据读取,将数据分为标签变量和特征
print("将数据拆分成特征和标签变量中...")
re_con_his = con_his.drop(["fraud_if"], axis=1)  # 特征
test_req = re_con_his
labels = con_his["fraud_if"]  # 标签变量

# 计算特征相关性，剔除相关性大的特征以降维
print("计算特征相关性中...")
corr_matrix = re_con_his.corr(method='spearman')
print("绘图中...")
plt.figure(figsize=(25, 15))
sns.heatmap(corr_matrix, annot=True)
plt.xticks(rotation=25)
plt.yticks(rotation=25)
plt.show()
print("绘图成功！")

# 存储相关性过高的特征对,对于相关性过高的的特征，删除其中一个（根据工程经验，以0.8为界）
print("检索相关性大于0.8的横纵坐标中...")
high_corr_pairs = np.column_stack(np.where(np.triu(np.abs(corr_matrix) > 0.8, k=1)))
print(high_corr_pairs)

# 丢弃特征对中的一个
print("选择特征对中的一个...")
cols_to_drop = re_con_his.columns[[col[1] for col in high_corr_pairs]]
print("丢弃特征:", cols_to_drop)
re_con_his = re_con_his.drop(cols_to_drop, axis=1)

print("读取数据中...")
print(re_con_his)


###　Step7：特征选取，数据降维
print('正在进行降维处理...')
print('特征维数小于等于5，不需要进行PCA降维处理！')

print("将特征选择结束的特征和标签变量合并中...")
con_his = pd.concat([re_con_his, labels], axis=1)
print('分析完成，输出经过PCA降维后的数据如下：')
print(con_his)

#### 客户信用记录预处理结束，将数据预处理结果导出为.csv文件
con_his.to_csv(r"file_processed\consumption_history_pro.csv")
print('客户信用记录预处理结束，将数据预处理结果导出为.csv文件,命名为：consumption_history_pro')



# ####  step8:使用随机森林模型进行训练
# 目标变量为信用总评分
y = labels

# 将数据集划分为训练集和测试集
print(' ')
print('正在划分训练集和测试集，其中测试集占0.2...')
X_train, X_test, y_train, y_test = train_test_split(re_con_his, y, test_size=0.2, random_state=42)
print('开始进行随机森林模型训练...')

# 训练一个随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 使用 RandomForestClassifier
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
print('在测试集上进行预测')
y_pred = rf_model.predict(X_test)

### 评估预测性能
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result_df = result_df.reset_index(drop=True)
print('开始进行模型准确度评估...')
accuracy = accuracy_score(y_test, y_pred)
print("随机森林模型准确度评估:", accuracy)

# 评估分类性能
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_rep)

# 查看预测结果与实际值之间的差异
result_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result_df = result_df.reset_index(drop=True)

print("\nPrediction Results:")
print(result_df)

# 获取特征重要性
print('绘制特征重要性直方图中...')
importance = rf_model.feature_importances_
df_importance = pd.DataFrame(importance, index=None, columns=["Importance"])
df_importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20, len(df_importance) / 2))
plt.title("Attribute significance")
plt.xlabel("significance")
plt.ylabel("Stats")
plt.show()
print('绘制完成！')

# 查看重要性突出的特征的分布
importance_df = df_importance.sort_values(by='Importance', ascending=False)
print("\nFeature Importance:")
print(importance_df)
fra_model = rf_model
print('模型建立结束！！')



####  step9:对随机森林模型进行交叉检验
## 首先通过交叉验证的方式检查样本分布不均是否对模型造成影响
cv_scores = cross_val_score(fra_model, X_train, y_train, scoring='accuracy', cv=3)
print("交叉验证检查样本分布不均是否对模型造成影响:\n", cv_scores)



#### step10:网格搜索参数优化
# 定义要搜索的参数范围
print('网格搜索开始...')
print('定义要搜索的参数范围中...')
param_grid = {
    'n_estimators': range(90, 200, 10),
    'max_depth': range(3, 20, 2)
}

# 执行网格搜索
print('执行网格搜索...')
grid_search = GridSearchCV(estimator=fra_model,
                           param_grid=param_grid,
                           scoring=make_scorer(fbeta_score, beta=1))

grid_search.fit(X_train, y_train)

# 打印最佳参数组合和得分
print('打印最佳参数组合和得分：')
print('Best parameters: {}'.format(grid_search.best_params_))
print('Best score: {}'.format(grid_search.best_score_))

# 保存最佳模型
joblib.dump(grid_search.best_estimator_, r"model\fra_model.pkl")

# 加载最佳模型
model_load = joblib.load(r"model\fra_model.pkl")

# 在测试集上进行预测
print('在测试集上进行预测：')
y_test_pred = model_load.predict(X_test)

# 计算并输出F1 score
print('F1 score of random forest regressor model: {}'.format(fbeta_score(y_test, y_test_pred, beta=1)))
