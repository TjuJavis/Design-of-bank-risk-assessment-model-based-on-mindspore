
######################################################################################################
# 请点击左下方打开终端（alt+F12），输入conda activate mindspore进入环境，再输入python Behavior_rating.py运行程序
######################################################################################################

#### 导入相关库
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

#### Step1：导入相关库和实验数据
req_cus = pd.read_csv("file_list/Request_customer_information.csv", index_col=0) #读取数据。index_col=0：读取时不自动添加行号。


#### 数据预处理
### 由于行为评级模型研究的是相关融资类业务中存量客户在续存期内的管理，因此删除以下两种情况的客户后再进行分析
print('删除没有贷款记录和现在没有贷款的客户，只考虑存量客户在续存期内的管理情况，建立行为评级模型...')
req_cus = req_cus.drop(req_cus[(req_cus['信贷情况'] == '没有贷款记录') | (req_cus['信贷情况'] == '现在没有贷款')].index)
req_cus.isnull().any()  # 当删除掉不需要的行时，行索引会变的不连续，这时候可以重新设计新的索引
req_cus.reset_index(drop=True, inplace=True)  # drop=True：删除原行索引；inplace=True:在数据上进行更新

### Step2：缺失值处理
print('判断是否含有缺失值:')
print(req_cus.isnull().sum())
# 发现req中只存在少量缺失数据，故可选择直接删除缺失数据行
print('发现req中只存在少量缺失数据，故直接删除缺失数据行。')
req_cus = req_cus.dropna()
req_cus.isnull().any()  # 当删除掉不需要的行时，行索引会变的不连续，这时候可以重新设计新的索引
req_cus.reset_index(drop=True, inplace=True)  # drop=True：删除原行索引；inplace=True:在数据上进行更新
# 检查处理后的数据集中是否含有缺失值
print(req_cus.isnull().any())
print('缺失值处理完成！')


### Step3：重复值处理
print('正在对重复值进行处理，即将删除多余的重复值，并更新索引...')
req_cus.drop_duplicates(inplace=True)
req_cus.reset_index(drop=True, inplace=True)
print('重复值处理完成！')


### Step4：异常值处理
# 查看数据基本特征
print('正在查看数据基本特征...')
print(req_cus.describe())

## 为便于后续的可视化及数据分析工作，对一些数据中与模型建立求解准确度显然无关的变量“客户号”、“客户姓名”、“户籍”、“额度”删去，
## 并将各列用英文表示
#  年龄 性别 	婚姻状态	教育程度	职业类别
#  居住类型	车辆情况  保险缴纳
#  工作年限	个人年收入 信贷情况
req_cus = req_cus.drop(columns=["客户姓名", "证件号码", "户籍"])
req_cus.columns = range(len(req_cus.columns))
req_cus.columns = ['age', 'sex', 'Marital_status', 'Educational_attainment', 'Occupation_category',
                   'Residential_type', 'Vehicle_condition', 'Insurance_payment',
                   'Working_years', 'Personal_income', 'Credit_position'
                   ]
print('数据中包含的特征（英文表示）为：')
print(req_cus.columns)

## 分析发现，req_cus中可能存在的异常值有两方面，即年龄和工作年限的大小关系、个人年收入的大小。故对其进行处理。
unreal0 = 0
for index, row in req_cus.iterrows():
    if row['age'] < row['Working_years']:
        print("发现异常值！年龄小于工作年限：", row['age'], "<", row['Working_years'])
        temp = row['age']
        req_cus.at[index, 'age'] = row['Working_years']
        req_cus.at[index, 'Working_years'] = temp
        unreal0 = unreal0 + 1

# 绘制散点图
# 读取req的前100行数据
print('正在绘制前100人个人年收入散点分布图')
plt.figure(figsize=(10, 6))
plt.scatter(req_cus.head(100).index - 1, req_cus.head(100)['Personal_income'])
plt.title('前100人个人年收入散点分布图')
plt.xlabel('客户序号')
plt.ylabel('个人年收入')
plt.show()
print('绘制完成！发现有明显的异常点！下面用数据展示个人年收入情况：')
print(req_cus.head(100)['Personal_income'].to_frame().describe())
print('发现中位值和平均值偏差太大！')

# 计算中位数和标准差
median = req_cus.head(100)['Personal_income'].median()
std = req_cus.head(100)['Personal_income'].std()

# 确定需要替换的异常值索引
outlier_index = req_cus.head(100)[req_cus.head(100)['Personal_income'] > median + 0.1*std].index

# 遍历异常值索引，将其替换为与其相邻的正常数据
unreal1 = 0
for idx in outlier_index:
    diff = np.abs(req_cus.head(100).loc[idx, 'Personal_income'] - median)
    replacement = req_cus.head(100)[(np.abs(req_cus.head(100)['Personal_income'] - median) < 0.1*std) & (req_cus.head(100).index != idx)]['Personal_income'].values
    if len(replacement) > 0:
        print('发现异常值！个人年收入过高：', req_cus.head(100).at[idx, 'Personal_income'])
        req_cus.head(100).at[idx, 'Personal_income'] = replacement[0]
        unreal1 = unreal1 + 1
unreal0 = unreal0 + unreal1
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
plot_distribution(req_cus, cols=5, width=30, height=30, hspace=0.55, wspace=0.5)
plt.show()
print("绘制完成!")


### Step5：数据编码
print('正在进行数据编码...')
# 对二分类数据采用 01 编码
binary_features = ['sex', 'Insurance_payment', 'Vehicle_condition']
for feature in binary_features:
    req_cus[feature] = req_cus[feature].map({'男': 1, '有': 1, '有': 1, '女': 0, '无': 0, '无': 0})

# 对有多值且相互独立的数据采用One-Hot编码
multi_value_features = ['Marital_status',  'Residential_type', 'Occupation_category']
req_cus = pd.get_dummies(req_cus, columns=multi_value_features)

# 对有序递进的数据采用Label Encoder编码
label_encoder = LabelEncoder()
req_cus['Educational_attainment'] = label_encoder.fit_transform(req_cus['Educational_attainment'])
req_cus['Credit_position'] = label_encoder.fit_transform(req_cus['Credit_position'])
# 将 Credit_position 中被 LabelEncoder 编号为 0 和 1 的重新编号为 0，编号为 2 和 3 的重新编为 1
# 新编号中，1 代表“还在拖欠”或“逾期还款”；0表示“正常还款”或“正在偿还”
req_cus['Credit_position'] = req_cus['Credit_position'].map({0: 0, 1: 0, 2: 1, 3: 1})

# 输出编码后的数据
print('编码完成！下面展示编码后的数据：')
print(req_cus)




### Step6：相关性分析
# 数据读取,将数据分为标签变量和特征
print("将数据拆分成特征和标签变量中...")
re_req_cus = req_cus.drop(["Credit_position"], axis=1)  # 特征
test_req = re_req_cus
labels = req_cus["Credit_position"]  # 标签变量

# 计算特征相关性，剔除相关性大的特征以降维
print("计算特征相关性中...")
corr_matrix = re_req_cus.corr(method='spearman')
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
cols_to_drop = re_req_cus.columns[[col[1] for col in high_corr_pairs]]
print("丢弃特征:", cols_to_drop)
re_req_cus = re_req_cus.drop(cols_to_drop, axis=1)

print("读取数据中...")
print(re_req_cus)


# 填充缺失值
re_req_cus.fillna(0, inplace=True)  # 使用0填充缺失值，也可以根据业务需求填充其他数值
# 删除包含缺失值的行
re_req_cus.dropna(inplace=True)
# 替换无穷大为特定数值
re_req_cus.replace([np.inf, -np.inf], np.nan, inplace=True)  # 将无穷大替换为NaN，然后再进行填充或删除操作
# 数据类型转换
data = re_req_cus.astype('float64')  # 将数据转换为float64类型


###　Step7：特征选取，数据降维
# 对数据进行标准化处理
print('数据标准化中...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(re_req_cus)
print('正在构建PCA对象，设置主成分数目为5...')
# 构建PCA对象，设置主成分数目为k
k = 5
pca = PCA(n_components=k)
print('正在进行PCA分析...')
# 对标准化后的数据进行PCA分析
pca.fit(X_scaled)
# 获取转换后的数据
X_pca = pca.transform(X_scaled)
print("将特征选择结束的特征和标签变量合并中...")
X_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5"])  # 将X_pca转换为DataFrame，并指定列名
re_req_cus = pd.concat([X_pca, labels], axis=1)
print('分析完成，输出经过PCA降维后的数据如下：')
print(re_req_cus)
#### 客户信用记录预处理结束，将数据预处理结果导出为.csv文件
re_req_cus.to_csv(r"file_processed\request_customer_pro.csv")
print('客户信用记录预处理结束，将数据预处理结果导出为.csv文件,命名为：request_customer_pro')



# ####  step8:使用随机森林模型进行训练
# 目标变量为信用总评分
y = labels

# 将数据集划分为训练集和测试集
print(' ')
print('正在划分训练集和测试集，其中测试集占0.2...')
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
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
beh_model = rf_model
print('模型建立结束！！')



####  step9:对随机森林模型进行交叉检验
## 首先通过交叉验证的方式检查样本分布不均是否对模型造成影响
cv_scores = cross_val_score(beh_model, X_train, y_train, scoring='accuracy', cv=3)
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
grid_search = GridSearchCV(estimator=beh_model,
                           param_grid=param_grid,
                           scoring=make_scorer(fbeta_score, beta=1))

grid_search.fit(X_train, y_train)

# 打印最佳参数组合和得分
print('打印最佳参数组合和得分：')
print('Best parameters: {}'.format(grid_search.best_params_))
print('Best score: {}'.format(grid_search.best_score_))

# 保存最佳模型
joblib.dump(grid_search.best_estimator_, r"model\beh_model.pkl")

# 加载最佳模型
model_load = joblib.load(r"model\beh_model.pkl")

# 在测试集上进行预测
print('在测试集上进行预测：')
y_test_pred = model_load.predict(X_test)

# 计算并输出F1 score
print('F1 score of random forest regressor model: {}'.format(fbeta_score(y_test, y_test_pred, beta=1)))
