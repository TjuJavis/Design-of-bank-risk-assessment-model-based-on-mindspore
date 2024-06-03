
######################################################################################################
# 请点击左下方打开终端（alt+F12），输入conda activate mindspore进入环境，再输入python applicant_rating.py运行程序
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

#### Step1：导入相关库和实验数据
cus_cre = pd.read_csv("file_list/Customer_credit_data.csv", index_col=0) #读取数据。index_col=0：读取时不自动添加行号。


#### 数据预处理
### Step2：缺失值处理
print('判断是否含有缺失值:')
print(cus_cre.isnull().sum())  # 发现customer_credit_data中没有缺失值，故不做处理
print('无缺失值，不做处理。')


### Step3：重复值处理
print('正在对重复值进行处理，即将删除多余的重复值，并更新索引...')
cus_cre.drop_duplicates(inplace=True)
cus_cre.reset_index(drop=True, inplace=True)
print('重复值处理完成！')


### Step4：异常值处理
# 查看数据基本特征
print('正在查看数据基本特征...')
print(cus_cre.describe())

## 为便于后续的可视化及数据分析工作，对一些数据中与模型建立求解准确度显然无关的变量“客户号”、“客户姓名”、“户籍”、“额度”删去，
## 并将各列用英文表示
#  性别 年龄	婚姻状态	教育程度	居住类型
#  职业类别	工作年限	个人收入	保险缴纳
#  车辆情况  信用总评分	信用等级	审批结果
cus_cre = cus_cre.drop(columns=[ "客户姓名", "户籍", "额度"])
cus_cre.columns = range(len(cus_cre.columns))
cus_cre.columns = ['sex', 'age', 'Marital_status', 'Educational_attainment', 'Residential_type',
                   'Occupation_category', 'Working_years', 'Personal_income', 'Insurance_payment',
                   'Vehicle_condition', 'Total_credit_score', 'Credit_rating', 'Approve_results'
                   ]
print('数据中包含的特征（英文表示）为：')
print(cus_cre.columns)

## 分析发现，cus_cre中可能存在的异常值仅一方面，即年龄和工作年限的大小关系，故对其进行处理
unreal0 = 0
for index, row in cus_cre.iterrows():
    if row['age'] < row['Working_years']:
        print("发现异常值！年龄小于工作年限：", row['age'], "<", row['Working_years'])
        temp = row['age']
        cus_cre.at[index, 'age'] = row['Working_years']
        cus_cre.at[index, 'Working_years'] = temp
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
plot_distribution(cus_cre, cols=5, width=30, height=30, hspace=0.55, wspace=0.5)
plt.show()
print("绘制完成!")


### Step5：数据编码
print('正在进行数据编码...')
# 对二分类数据采用 01 编码
binary_features = ['sex', 'Approve_results', 'Insurance_payment', 'Vehicle_condition']
for feature in binary_features:
    cus_cre[feature] = cus_cre[feature].map({'有': 1, '通过': 1, '有': 1, '男': 1, '无': 0, '未通过': 0, '无': 0, '女': 0})

# 对有多值且相互独立的数据采用One-Hot编码
multi_value_features = ['Marital_status',  'Residential_type', 'Occupation_category', 'Credit_rating']
cus_cre = pd.get_dummies(cus_cre, columns=multi_value_features)

# 对有序递进的数据采用Label Encoder编码
label_encoder = LabelEncoder()
cus_cre['Educational_attainment'] = label_encoder.fit_transform(cus_cre['Educational_attainment'])
print('编码完成！下面展示编码后的数据：')
print(cus_cre)


### Step6：相关性分析
# 数据读取,将数据分为标签变量和特征
print("将数据拆分成特征和标签变量中...")
re_cus_cre = cus_cre.drop(["Total_credit_score"], axis=1)  # 特征
labels = cus_cre["Total_credit_score"]  # 标签变量

# 计算特征相关性，剔除相关性大的特征以降维
print("计算特征相关性中...")
corr_matrix = re_cus_cre.corr(method='spearman')
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
cols_to_drop = re_cus_cre.columns[[col[1] for col in high_corr_pairs]]
print("丢弃特征:", cols_to_drop)
re_cus_cre = re_cus_cre.drop(cols_to_drop, axis=1)

print("读取数据中...")
print(re_cus_cre)


# 填充缺失值
re_cus_cre.fillna(0, inplace=True)  # 使用0填充缺失值，也可以根据业务需求填充其他数值
# 删除包含缺失值的行
re_cus_cre.dropna(inplace=True)
# 替换无穷大为特定数值
re_cus_cre.replace([np.inf, -np.inf], np.nan, inplace=True)  # 将无穷大替换为NaN，然后再进行填充或删除操作
# 数据类型转换
data = re_cus_cre.astype('float64')  # 将数据转换为float64类型


###　Step7：特征选取，数据降维
# 对数据进行标准化处理
print('数据标准化中...')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(re_cus_cre)
print('正在构建PCA对象，设置主成分数目为6...')
# 构建PCA对象，设置主成分数目为k
k = 6
pca = PCA(n_components=k)
print('正在进行PCA分析...')
# 对标准化后的数据进行PCA分析
pca.fit(X_scaled)
# 获取转换后的数据
X_pca = pca.transform(X_scaled)
print("将特征选择结束的特征和标签变量合并中...")
X_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6"])  # 将X_pca转换为DataFrame，并指定列名
cus_cre_pca = pd.concat([X_pca, labels], axis=1)
print('分析完成，输出经过PCA降维后的数据如下：')
print(cus_cre_pca)
#### 客户信用记录预处理结束，将数据预处理结果导出为.csv文件
cus_cre_pca.to_csv(r"file_processed\customer_credit_pro.csv")
print('客户信用记录预处理结束，将数据预处理结果导出为.csv文件,命名为：customer_credit_pro')



####  step8:使用随机森林模型进行训练
# 目标变量为信用总评分
y = labels
# 将数据集划分为训练集和测试集
print(' ')
print('正在划分训练集和测试集，其中测试集占0.2...')
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
print('开始进行随机森林模型训练...')
# 训练一个随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
print('在测试集上进行预测')
y_pred = rf_model.predict(X_test)

### 评估预测性能
print('对预测到的分数进行评级A(90~100)、B(80~90)、C(70~80)、D(60~70)...')
def map_to_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    else:
        return 'D'

y_test_grade = [map_to_grade(score) for score in y_test]
y_pred_grade = [map_to_grade(score) for score in y_pred]
result_df = pd.DataFrame({'Actual': y_test_grade, 'Predicted': y_pred_grade})
result_df = result_df.reset_index(drop=True)
print('开始进行模型准确度评估...')
accuracy = (result_df['Actual'] == result_df['Predicted']).mean()
f1 = f1_score(y_test_grade, y_pred_grade, average='weighted')
print("随机森林模型准确度评估:", accuracy)
# 查看预测结果与实际值之间的差异
print("\n实际值与预测值数据对比:")
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
print("查看重要性突出的特征的分布:")
for feature in df_importance[df_importance['Importance'] > 0.05].index:
    print(X_pca.iloc[:, feature].value_counts())
apl_model = rf_model
print('模型建立结束！！')


####  step9:对随机森林模型进行交叉检验
## 首先通过交叉验证的方式检查样本分布不均是否对模型造成影响
# 进行交叉验证并计算模型的均方误差（MSE）
print('开始进行交叉检验...')
scores = cross_val_score(apl_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# 将MSE转化为RMSE
rmse_scores = [-score for score in scores]
rmse_scores = np.sqrt(rmse_scores)
print("随机森林模型交叉验证RMSE:", rmse_scores.mean())


####  step10:网格搜索参数优化
print("网格搜索开始")
# 利用坐标下降的方式，对t模型的关键参数进行搜索，尝试获取更好的建模结果
# 首先对参数 n_estimators 进行搜索
param_test1 = {'n_estimators': range(10, 200, 5)}
gsearch1 = GridSearchCV(estimator=apl_model,
                        param_grid=param_test1,
                        scoring=make_scorer(r2_score))
gsearch1.fit(X_train, y_train)
print('best params:{}'.format(gsearch1.best_params_))
print('best score:{}'.format(gsearch1.best_score_))

# 对参数 max_depth 进行搜索
print("对参数 max_depth 进行搜索...")
param_test2 = {'max_depth': range(0, 10, 1)}
gsearch2 = GridSearchCV(estimator=gsearch1.best_estimator_,
                        param_grid=param_test2,
                        scoring=make_scorer(r2_score))
gsearch2.fit(X_train, y_train)
print('best params2:{}'.format(gsearch2.best_params_))
print('best score:{}'.format(gsearch2.best_score_))
print(joblib.dump(gsearch2.best_estimator_, r"model\apl_model.pkl"))
model_load = joblib.load(r"model\apl_model.pkl")
y_test_pred = model_load.predict(X_test)
print('r2 score of random forest score:{}'.format(r2_score(y_test, y_test_pred)))
print("优化结束！")





