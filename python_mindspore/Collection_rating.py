
######################################################################################################
# 请点击左下方打开终端（alt+F12），输入conda activate mindspore进入环境，再输入python Collection_rating.py运行程序
######################################################################################################

#### 导入相关库
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定字体为SimHei

#### Step1：导入相关库和实验数据
def_his = pd.read_csv("file_list/Default history.csv", index_col=0) #读取数据。index_col=0：读取时不自动添加行号。


#### 数据预处理
### Step2：缺失值处理
print('判断是否含有缺失值中...')
print(def_his.isnull().sum())
print('未发现缺失值，处理完毕！')


### Step3：重复值处理
print('正在对重复值进行处理，即将删除多余的重复值，并更新索引...')
def_his.drop_duplicates(inplace=True)
def_his.reset_index(drop=True, inplace=True)
print('重复值处理完成！')


### Step4：异常值处理
# 查看数据基本特征
print('正在查看数据基本特征...')
print(def_his.describe())

## 为便于后续的可视化及数据分析工作，对一些数据中与模型建立求解准确度显然无关的变量"卡类别", "币种代码", "额度"删去，
## 并将各列用英文表示
#  额度	拖欠标识	拖欠总金额	逾期天数
def_his = def_his.drop(columns=["卡号"])
def_his.columns = range(len(def_his.columns))
def_his.columns = ['limit', 'default_marking', 'total_arrears', 'days_overdue']
print('数据中包含的特征（英文表示）为：')
print(def_his.columns)

## 分析发现，def_his种不存在异常值。
print('正在查找可疑异常值，请稍等...')
print('未发现任何异常值！')

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
plot_distribution(def_his, cols=5, width=30, height=30, hspace=0.55, wspace=0.5)
plt.show()
print("绘制完成!")


### Step5：数据编码
print('正在进行数据编码...')
print('数据均为数字，未发现需编码数据！')
print(def_his)


### Step6：相关性分析
# 数据读取,将数据分为标签变量和特征
print("将数据拆分成特征和标签变量中...")
re_def_his = def_his.drop(["days_overdue"], axis=1)  # 特征
test_def = re_def_his
labels = def_his["days_overdue"]  # 标签变量

# 计算特征相关性，剔除相关性大的特征以降维
print("计算特征相关性中...")
corr_matrix = re_def_his.corr(method='spearman')
print('spearman相关系数展示如下：')
print(corr_matrix)
print("检索相关性大于0.8的横纵坐标中...")
print('由于特征维度较低且关系直观。因此可直接进行手动降维，去掉“额度”和“拖欠标识”。')
print("读取数据中...")
re_def_his = re_def_his.drop(["limit", "default_marking"], axis=1)  # 特征
print(re_def_his)

#"limit",
###　Step7：特征选取，数据降维
print('正在进行降维处理...')
print('特征维数小于等于5，不需要进行PCA降维处理！')

print("将特征选择结束的特征和标签变量合并中...")
def_his = pd.concat([re_def_his, labels], axis=1)
print('分析完成，输出经过PCA降维后的数据如下：')
print(def_his)

#### 客户信用记录预处理结束，将数据预处理结果导出为.csv文件
def_his.to_csv(r"file_processed\default_history_pro.csv")
print('客户信用记录预处理结束，将数据预处理结果导出为.csv文件,命名为：default_history_pro.csv')


####  step8:使用随机森林模型进行训练
# 目标变量为信用总评分
y = labels
# 将数据集划分为训练集和测试集
print(' ')
print('正在划分训练集和测试集，其中测试集占0.2...')
X_train, X_test, y_train, y_test = train_test_split(re_def_his, y, test_size=0.2, random_state=42)
print('开始进行随机森林模型训练...')
# 训练一个随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
print('在测试集上进行预测')
y_pred = rf_model.predict(X_test)

### 评估预测性能
print('对预测到的天数用进行评级B(>=80)、A(0~80)...')
def map_to_grade(score):
    if score >= 80:
        return 'B:要加大催收力度和频率，甚至采取强制措施'
    else:
        return 'A：有规律地进行催收提醒，保持节奏'

y_test_grade = [map_to_grade(score) for score in y_test]
y_pred_grade = [map_to_grade(score) for score in y_pred]
result_df = pd.DataFrame({'Actual': y_test_grade, 'Predicted': y_pred_grade})
result_df = result_df.reset_index(drop=True)
print('开始进行模型准确度评估...')
accuracy = f1_score(y_test_grade, y_pred_grade, average='weighted')
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
    print(re_def_his.iloc[:, feature].value_counts())
dfa_model = rf_model
print('模型建立结束！！')


####  step9:对随机森林模型进行交叉检验
## 首先通过交叉验证的方式检查样本分布不均是否对模型造成影响
# 进行交叉验证并计算模型的均方误差（MSE）
print('开始进行交叉检验...')
scores = cross_val_score(dfa_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# 将MSE转化为RMSE
rmse_scores = [-score for score in scores]
rmse_scores = np.sqrt(rmse_scores)
print("随机森林模型交叉验证RMSE:", rmse_scores.mean())


####  step10:网格搜索参数优化
print("网格搜索开始...")
# 利用坐标下降的方式，对t模型的关键参数进行搜索，尝试获取更好的建模结果
# 首先对参数 n_estimators 进行搜索
print("对参数 n_estimators 进行搜索...")
param_test1 = {'n_estimators': range(10, 200, 5)}
gsearch1 = GridSearchCV(estimator=dfa_model,
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
print(joblib.dump(gsearch2.best_estimator_, r"model\dfa_model.pkl"))
model_load = joblib.load(r"model\dfa_model.pkl")
y_test_pred = model_load.predict(X_test)
print('r2 score of random forest score:{}'.format(r2_score(y_test, y_test_pred)))
print("优化结束！")
