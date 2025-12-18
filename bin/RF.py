#!/usr/bin/env python

"""
RF
===============================================
Author: Chen Jing (cjwell@163.com)
"""
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict,learning_curve,StratifiedKFold,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.impute import SimpleImputer  ##缺失值处理
import joblib



##bar图
def plot_top_bars(df, col1, col2, col3, top_n, colors,output_file):
    # 确保传入的列名存在于数据框中
    if col1 not in df.columns or col2 not in df.columns or col3 not in df.columns:
        raise ValueError("指定的列名在数据框中不存在")

    # 根据列2的值从高到低排序，并选择前top_n个
    sorted_df = df.nlargest(top_n, col2)

    # 获取唯一分类变量和颜色映射
    unique_categories = sorted_df[col3].unique()
    color_map = {cat: colors[i] for i, cat in enumerate(sorted(unique_categories))}
    
    # 如果分类变量数量超过颜色队列长度，使用灰色
    default_color = 'gray'
    sorted_df['color'] = sorted_df[col3].map(color_map).fillna(default_color)

    # 绘制条形图
    plt.figure(figsize=(10, 10))
    plt.barh(sorted_df[col1], sorted_df[col2], color=sorted_df['color'],edgecolor='black')
    plt.xlabel('Values of ' + col2)
    plt.ylabel(col1)
    plt.title(f'Top {top_n} {col1} by {col2}')
    plt.gca().invert_yaxis()  # 使得值高的条形在上方
     # 调整子图参数以保留图例空间
    plt.subplots_adjust(right=0.85)  # 增加右边的空白,值需要大于left
    plt.subplots_adjust(left=0.4)  # 增大左边空白，值可以根据需要调整

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[cat], edgecolor='black') for cat in unique_categories]
    plt.legend(handles, unique_categories, title='', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(output_file, format='pdf')
    plt.close()


#学习曲线
def plot_learning_curve(estimator,X, y, output_pdf,score="accuracy",title=None): ##模型，x，y，输出文件，评分方法，标题
    # 构建随机森林分类器
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    # 设置5折交叉验证
    #cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv=StratifiedKFold(n_splits=5)
    estimator.fit(X,y)
    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=score, n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5))

    # 绘制学习曲线
    plt.figure()
    if title is None:
        title = "Learning Curves (Random Forest)"
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    ylim=(0.5, 1.01)
    if ylim is not None:
        plt.ylim(*ylim)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    # 保存学习曲线图像
    plt.savefig(output_pdf, format='pdf')
    plt.close()
    print(f"Learning curve saved to {output_pdf}")

def plot_cv_roc(model, X, y, cv, target_class, output_pdf):
    """
    该函数使用指定的模型进行K折交叉验证，并绘制train和test的ROC曲线，标注AUC平均值和标准差。

    参数:
    - model: 训练好的模型
    - X: 观察变量 (特征)
    - y: 预测变量 (标签)
    - cv: 交叉验证折数
    - target_class: 要预测的目标类
    - output_pdf: 输出文件路径
    """
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X,y)

    ##5折拆分
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gn=y.name
    train_aucs = []
    test_aucs = []

    plt.figure(figsize=(8, 8))

    
    # 创建一个 DataFrame
    data = pd.DataFrame(columns=['Index', 'TrainAUC', 'TestAUC',"Target"])
    print(data)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        
        # 切分训练集和测试集
        X_train, X_test = X.iloc[train_index, 1:-1] ,X.iloc[test_index, 1:-1] 
        y_train, y_test = y[train_index], y[test_index]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print("Skipping fold due to class imbalance")
            continue

        # 训练模型
        model.fit(X_train, y_train)
        # 获取target_class在model.classes_中的索引

        target_class_index = np.where(model.classes_ == target_class)[0][0]
	##强制二分类绘制ROC
        ty_train,ty_test=y_train,y_test
        ty_train = ty_train.apply(lambda x: x if x == target_class else 'other')
        ty_test = ty_test.apply(lambda x: x if x == target_class else 'other')


        # 预测训练集和测试集的概率
        y_train_scores = model.predict_proba(X_train)[:, target_class_index]
        y_test_scores = model.predict_proba(X_test)[:, target_class_index]
     
        # 计算train的ROC曲线和AUC
        fpr_train, tpr_train, _ = roc_curve(ty_train, y_train_scores, pos_label=target_class)
        roc_auc_train = auc(fpr_train, tpr_train)
        train_aucs.append(roc_auc_train)

        # 计算test的ROC曲线和AUC
        fpr_test, tpr_test, _ = roc_curve(ty_test, y_test_scores, pos_label=target_class)
        roc_auc_test = auc(fpr_test, tpr_test)
        test_aucs.append(roc_auc_test)

        ##记录AUC
        data.loc[i] = [i+1, roc_auc_train, roc_auc_test,target_class]

        # 绘制每折的train和test ROC曲线
        plt.plot(fpr_train, tpr_train, lw=1.5, label=f'Train ROC fold {i+1} (AUC = {roc_auc_train:.2f})')
        plt.plot(fpr_test, tpr_test, lw=1.5, label=f'Test ROC fold {i+1} (AUC = {roc_auc_test:.2f})', linestyle='--')
        plt.legend(loc='lower right',fontsize=16)     
    # 计算平均值和标准差
    mean_train_auc = np.mean(train_aucs)
    std_train_auc = np.std(train_aucs)
    
    mean_test_auc = np.mean(test_aucs)
    std_test_auc = np.std(test_aucs)

    data.loc[i+1] = ['mean', mean_train_auc, mean_test_auc,target_class]
    data.loc[i+2] = ['std', std_train_auc, std_test_auc,target_class]
    df = pd.DataFrame(data)

    # 图表设置
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title(f'ROC Curve with {cv}-fold CV (Positive Class: {target_class} of {gn})')
    
    # 标注AUC平均值和标准差
    plt.legend(loc='lower right', title=f'Train AUC = {mean_train_auc:.2f} ± {std_train_auc:.2f}\n'
                                        f'Test AUC = {mean_test_auc:.2f} ± {std_test_auc:.2f}')
    
    # 保存图像
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"ROC曲线已保存至: {output_pdf}")
    return df

def plot_roc_curve(model, x_train,x_test,y_train,y_test,target_class, output_pdf):
    """
    生成并保存指定类别的ROC曲线。

    参数:
    - model: 训练好的随机森林模型
    - x_train: 训练观察变量
    - y_train: 训练预测分类变量
    - x_test: 测试观察变量
    - y_test: 测试预测分类变量
    - target_class: 指定的目标类别（y_train中的某一类别）
    - output_pdf: 输出文件路径
    """

    # 获取target_class在classes_中的索引
    classes = model.classes_
    gn=y_train.name
    target_class_index = np.where(classes == target_class)[0][0]

    # 预测训练集和测试集的概率
    y_train_scores = model.predict_proba(x_train)[:, target_class_index]   # 选择正类的概率
    y_test_scores = model.predict_proba(x_test)[:, target_class_index]  

    # 计算ROC曲线和AUC
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_scores, pos_label=target_class)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_scores, pos_label=target_class)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_test = auc(fpr_test, tpr_test)

    # 创建一个 DataFrame
    data = pd.DataFrame(columns=['Index', 'TrainAUC', 'TestAUC','Target'])
    data.loc[1] = ['final', roc_auc_train, roc_auc_test,target_class]
    df = pd.DataFrame(data)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_train, tpr_train, color='blue', lw=3,label=f'Train (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='red', lw=3,label=f'Test  (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 随机猜测的基准线

    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title(f'ROC Curve for Class {target_class} of {gn}')
    plt.legend(fontsize=16,loc="lower right") #8的2倍
    plt.grid()

    # 保存图像
    plt.savefig(output_pdf, format='pdf')
    plt.close()
    print(f'Final ROC 曲线保存在 {output_pdf}')
    return df


def df_roc_curve(df,sb,actual,tg,output_pdf):
    """
    生成并保存指定类别的ROC曲线。
    参数:
    - tg: 指定的目标类别（df中的某一类别）
    - output_file: 输出文件全路径

    """
    # 绘制ROC曲线
    plt.figure(figsize=(8, 8))
    auc_df=[]
    if tg is None:
        tg=df.columns[1]
    for target in df[sb].unique():
            if target is None:
                continue
            #子数据框
            sub_df = pd.DataFrame()    
            sub_df=df.copy()
            sub=sb
            
            if sb == actual:
                ##新增一列
                sub_df["sub"]=target
                sub="sub"
                target_class=target
                # 替换actual中不是target的值为'other'
                sub_df.loc[sub_df[actual] != target, actual] = 'other'
            else:
                #取子集
                sub_df = sub_df[sub_df[sub] ==target]
                target_class=tg

            ##子数据框的actual
            y=sub_df[actual]
            y_scores=sub_df[target_class]

            fpr, tpr, _ = roc_curve(y, y_scores, pos_label=target_class)
            roc_auc=auc(fpr,tpr)
            auc_df.append([target,target_class,roc_auc])
            plt.plot(fpr, tpr, lw=3,label=f'{target} (AUC = {roc_auc:.2f})')
            plt.legend(loc='lower right',fontsize=16)   
    # 图表设置
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title(f'ROC Curve')
    plt.grid()

    # 保存图像
    plt.savefig(output_pdf, format='pdf')
    plt.close()
    print(f'ROC 曲线保存在 {output_pdf}')
    ##auc输出
    auc_df = pd.DataFrame(auc_df, columns=['run', 'target', 'AUC'])
    return auc_df

def optimize_random_forest(X, y, output_path,file_prefix,colors=["red","blue","green"],med="grid", top_n=10):
    """
    优化随机森林模型的超参数，并返回最佳模型和超参数，保存特征信息。
    
    参数：
    X: 特征矩阵 (numpy array 或 pandas DataFrame)
    y: 目标变量 (numpy array 或 pandas Series)
    med: 网格搜索方法 ("grid" or "random")
    top_n:绘制top_n的特征重要性
    output_path：输出路径
    file_prefix：输出文件前缀
    返回值：
    best_rf: 优化后的随机森林模型

    """

    # 定义要优化的超参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],    # 树的数量
        'max_depth': [None, 10, 20],       # 每棵树的最大深度
        'min_samples_split': [2, 5, 10],   # 分裂节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4],     # 叶节点所需的最小样本数
        'max_features': ['auto', 'sqrt']   # 最大特征数
    }
 
    # 初始化随机森林分类器
    rf = RandomForestClassifier(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 方法1: 网格搜索
    if med == "grid":
        grid_search = GridSearchCV(estimator=rf,param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_

    # 方法2: 随机搜索
    elif med == "random":
        random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
        random_search.fit(X, y)
        best_rf = random_search.best_estimator_
        best_params = random_search.best_params_
    # 方法3: 基础模型
    else:
        best_rf=RandomForestClassifier(n_estimators=100, random_state=42)
        best_rf=best_rf.fit(X_train,y_train) ##训练模型
        best_params=best_rf.get_params()
  
    #保存最佳参数
    output_file = os.path.join(output_path, f"{file_prefix}_best.params.txt")
    # 打开一个txt文件并保存字典
    with open(output_file, 'w', encoding='utf-8') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")  # 每对键值对后面加一个换行符
   
    # 提取特征重要性
    feature_importances_ = best_rf.feature_importances_

    # 如果 X 是 DataFrame，获取特征名
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # 排序特征重要性（降序）
    sorted_indices = np.argsort(feature_importances_)[::-1]
    sorted_importances = feature_importances_[sorted_indices]
    sorted_features=feature_names[sorted_indices]

    # 计算每个特征在 y 中每个分组的平均值
    grouped_means = X.groupby(y).mean()  # 按 y 的分组计算 X 的均值

    # 计算每个特征的最大均值及其对应的分组
    max_groups = grouped_means.idxmax()  # 找到最大均值的分组名
    max_values = grouped_means.max()      # 找到最大均值


    ##保存重要性
    output_file = os.path.join(output_path, f"{file_prefix}_importances.tsv")
    # 创建结果数据框
    results = pd.DataFrame({
    'features': feature_names,
    'importances': feature_importances_,
    'rich_group': max_groups,
    'rich_values': max_values 
    })

    df = pd.DataFrame(results)
    df.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符

    output_file = os.path.join(output_path, f"{file_prefix}_importances.pdf")
    plot_top_bars(df,"features","importances","rich_group",10,colors,output_file)
 
    # 将模型、特征和重要性保存到文件（如果提供了保存路径）
    output_file = os.path.join(output_path, f"{file_prefix}_model.pkl")

    joblib.dump({
    'model': best_rf,
    'feature_importances': feature_importances_,
    'feature_names': feature_names,
    'x': X,
    'y': y
    }, output_file)
    print(f"模型和特征信息已保存至 {output_file}")
    # 返回优化后的模型
    return best_rf,X_train, X_test, y_train, y_test 


def predict_scores(model, X,y,run="run"):
    
    # 预测各分类的预测分
    y_proba = model.predict_proba(X)
    # 获取最终的预测分类
    y_pred = model.predict(X)
    # 创建结果 DataFrame
    results = pd.DataFrame(y_proba, columns=model.classes_)
    results['sample_name'] = X.index

    results['predicted_class'] = y_pred
    results['run'] = run
    if y is not None:
        results['actual_class'] = y.tolist()
        # 调整列顺序
        results = results[['sample_name'] + list(model.classes_) + ['predicted_class']+['actual_class']+['run']]
    else:
        results = results[['sample_name'] + list(model.classes_) + ['predicted_class']+['run']]
    # 保存预测分
    df = pd.DataFrame(results)
    return df



def reorder_and_fill_features(index, X):
    """
    根据table中的特征对X进行排序，并填充X中没有的特征，值为0
    
    参数：
    table_path: 包含特征和重要性的tsv文件路径
    X: 观察变量的DataFrame
    
    返回：
    X_new: 处理后的DataFrame，包含table中的所有特征，缺失特征填充为0
    """
    if isinstance(index, pd.Index):
        features=index.tolist()
    elif isinstance(index, str) and os.path.isfile(index):
        # 读取table TSV文件
        table = pd.read_csv(index, sep='\t')
        # 提取table中的特征列表
        features = table['features'].tolist()
    else:
        return "Unknown type"
    
    
    # 初始化新的X_new，只保留X中存在于features中的列
    X_new = X[features].copy() if all(f in X.columns for f in features) else X.reindex(columns=features, fill_value=0)
    # 对于X中没有的特征，添加并填充为0
    for feature in features:
        if feature not in X.columns:
            X_new[feature] = 0
    
    # 确保列顺序和features保持一致
    X_new = X_new[features]
    
    return X_new

def onehot(X):
    # 确保数据类型正确
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype(str)  # 转换为字符串
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce')  # 转换为数值，无法转换的变为NaN

    # 自动识别分类变量
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols)==0:
        return X
    # 创建 OneHotEncoder 对象
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # 创建 ColumnTransformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', onehot_encoder, categorical_cols)  # 对识别的分类变量进行独热编码
        ],
        remainder='passthrough'  # 其他数值型变量保持不变
    )

    # 进行转换
    X_transformed = column_transformer.fit_transform(X)

    # 获取处理后的特征名称
    categorical_feature_names = column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = list(categorical_feature_names) + [col for col in X.columns if col not in categorical_cols]

    # 将 X_transformed 转换为 DataFrame，并指定列名
    X_transformed_df = pd.DataFrame(X_transformed, columns=all_feature_names)

    X_transformed_df.index =X.index
    return X_transformed_df

def main():
    parser = argparse.ArgumentParser(description="Random Forest Classification Model using -i for feature table and -g for binary phenotype grouping.")
    parser.add_argument("-i", "--feature_table", required=True, help="输入特征队列，column为feature，row为样本")
    parser.add_argument("-it", "--feature_transpose", action='store_true', help="当-i column为样本，row为feature时，颠置特征队列")
    #parser.add_argument("-oh", "--onehot", action='store_true', help="-i中含非数值变量时，设置-oh 对-i进行onehot编码")
    parser.add_argument("-g", "--phenotype", required=False, help="表型分组，第一列样本，第二列表型分组")
    parser.add_argument("-m", "--model", required=False, help="model file.pkl")
    parser.add_argument("-o", "--output", required=False, default="./" , help="Output directory")
    parser.add_argument("-p", "--prefix", required=False, default="RF" , help="File name prefix")
    parser.add_argument("-ta", "--target", required=False , default=None,help="指定绘制ROC曲线中预测的对象")
    parser.add_argument("-tb", "--table", required=False , help="输入imortances.tsv")
    parser.add_argument("-opt", "--optimize", required=False ,default="grid", help="超参数选择,可选grid，random和None")
    parser.add_argument("-cc", "--cf_counts", required=False , type=int,default=10,help="学习曲线和AUC.cv评价的最低样本数，过低的数量可能导致抽样问题")
    parser.add_argument("-bc", "--bar_colors", required=False,default=["red","blue","green"],type=str, nargs='+', help='bar图颜色的List,例如 -cl red blue green')    
    args = parser.parse_args()
    # 创建输出目录（如果不存在）
    os.makedirs(args.output, exist_ok=True)

    X = pd.read_csv(args.feature_table, sep='\t', index_col=0, dtype={'column_name': float},low_memory=False)
    if args.feature_transpose is not False:
	    X=X.T
    if args.phenotype is not None:
        # 读取表型分组信息文件
        y = pd.read_csv(args.phenotype, sep='\t', index_col=0,dtype={'column_name': str},low_memory=False)
        # 忽略第二列名字为空的行
        y = y.dropna(subset=[y.columns[0]])
        # 合并两个数据框，只保留共有的样本
        merged_data = pd.merge(X,y, left_index=True, right_index=True, how='inner')
        # 替换特征表格中的缺失值为0
        merged_data = merged_data.fillna(0)

        ##重新分配X，y
        X = merged_data.iloc[:, :-1]
        y = merged_data.iloc[:, -1]
        print("数据集shape：")
        print(merged_data.shape)
        if merged_data.shape[0]==0:
            print("-i -g交集为空")
            sys.exit()
        ##检验X是否存在分类变量
        X=onehot(X)

        ##检测-ta值
        unique_y=y.unique()
        if args.target is not None and args.target not in unique_y:
            print(f" -ta 取值 为 {unique_y[0]}")
            args.target=unique_y[0]
	
	# 构建随机森林分类器
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

	#预设model  -i -g -m
        if args.model is not None:
            model_dict= joblib.load(args.model)
            rf_model = model_dict['model']  # 获取模型
            X_train=model_dict['x']  # 获取模型x
            y_train=model_dict['y']  # 获取模型y
            X_test=X  # 获取输入x
            y_test=y  # 获取输入y
            df1=predict_scores(rf_model,X_train,y_train,"train")
            df2=predict_scores(rf_model,X_test,y_test,"validation")
            df = pd.concat([df1, df2], axis=0)
            # 保存为 tab 格式的文件
            output_file = os.path.join(args.output, f"{args.prefix}_validation.scores.tsv")
            df.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符
            ##绘制ROC曲线
            y_groups=y_train.unique()
            if len(y_groups) >5: ##超过5分类不执行
                print("-g 超过5个分类")
                sys.exit()
            if len(y_groups) ==2: ##二分类
                output_file = os.path.join(args.output, f"{args.prefix}_validation.roc.pdf")
                auc=df_roc_curve(df,"run","actual_class",args.target,output_file) ##训练集结果，验证集结果，二分类对象，输出pdf路径
                pd.DataFrame(auc)
                # 保存AUC
                output_file = os.path.join(args.output, f"{args.prefix}_validation.AUC.tsv")
                auc.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符         
            elif len(y_groups) >2: ##多分类
               auc_list=[]
               for y_target in y_groups:
                   if args.target is not None and args.target != y_target:
                    continue
                   ##绘制ROC_cv曲线
                   output_file = os.path.join(args.output, f"{args.prefix}_validation.{y_target}.roc.pdf")
                   auc_tmp=df_roc_curve(df,"run","actual_class",args.target,output_file) ##训练集结果，验证集结果，二分类对象，输出pdf路径
                   auc_list.append(auc_tmp)     
               # 保存过程AUC
               auc=pd.concat(auc_list,ignore_index=True) ##垂直合并
               output_file = os.path.join(args.output, f"{args.prefix}_validation.AUC.tsv")
               auc.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符

	##重新计算 -i -g
        else:
            #"""
            ##记录分布
            distribution = y.value_counts()
            output_file = os.path.join(args.output, f"{args.prefix}_train.distribution.txt")
            distribution.to_csv(output_file, sep='\t', header=True)    

            # 调用函数进行超参数优化,并保存模型重要性特征等
            rf_model,X_train, X_test, y_train, y_test = optimize_random_forest(X, y,args.output,args.prefix,args.bar_colors,args.optimize,10) #X,y,输出路径，输出前缀，方法,,top
 
            ##保存预测分
            df1=predict_scores(rf_model,X_train,y_train,"train")
            df2=predict_scores(rf_model,X_test,y_test,"test")
            df = pd.concat([df1, df2], axis=0)

            # 保存为 tab 格式的文件
            output_file = os.path.join(args.output, f"{args.prefix}_train.scores.tsv")
            df.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符
            output_file = os.path.join(args.output, f"{args.prefix}_train.roc.pdf")
            auc=df_roc_curve(df,"actual_class","actual_class",args.target,output_file) ## df，subset,actual,target,输出pdf路径
            pd.DataFrame(auc)
            # 保存AUC
            output_file = os.path.join(args.output, f"{args.prefix}_train.AUC.tsv")
            auc.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符  
            #"""
            y_groups=y.unique() ##分类数目
            if len(y_groups) >5: ##超过5分类不执行
                print("-g 超过5个分类")
                sys.exit()
            if len(y_groups) ==2:
                print("进行二分类评价")
                ty=y.copy()
                # 绘制学习曲线
                output_pdf=os.path.join(args.output, f"{args.prefix}_LearningCurves.pdf")
                plot_learning_curve(None,X,ty,output_pdf,"accuracy")

            elif len(y_groups) >2:
                print("进行多分类评价")
                for y_target in y_groups:
                    if args.target is not None and args.target != y_target:
                        continue
                    ##数目限制
                    yn=(y == y_target).sum()
                    if yn <args.cf_counts:
                        print(f"{y_target}分组仅{yn}个，低于{args.cf_counts}，跳过学习曲线评价")	          
                        continue		    
                    # 绘制学习曲线
                    output_pdf=os.path.join(args.output, f"{args.prefix}_LearningCurves.{y_target}.pdf")
                    ty=y.copy()
                    ty = ty.apply(lambda x: x if x == y_target else 'other')   ##替换为二分类

                    title=f"Learning Curves (Positive Class: {y_target} of {y.name})"
                    score="accuracy"
                    plot_learning_curve(None,X,ty,output_pdf,score,title) ##模型，x，y，输出文件，评分方法(roc_auc_ovr,roc_auc,accuracy)，标题

            ##绘制cv ROC曲线
            auc_list=[]             
            for y_target in y_groups:
                if args.target is not None and args.target != y_target:
                    continue
                ##数目限制
                yn=(y == y_target).sum()

                if yn <args.cf_counts:
                    print(f"{y_target}分组仅{yn}个，低于{args.cf_counts}，跳过评价")	          
                    continue			    


                ##绘制5折ROC_cv曲线
                ty=y.copy()
                output_file=os.path.join(args.output, f"{args.prefix}_train.{y_target}.roc.cv.pdf")
                auc_tmp=plot_cv_roc(None,X,ty,5,y_target,output_file)
                print(auc_tmp)
                ##记录每个分类的AUC
                auc_list.append(auc_tmp)   

            # 保存全部过程AUC
            cvauc=pd.concat(auc_list,ignore_index=True) ##垂直合并
            output_file = os.path.join(args.output, f"{args.prefix}_train.AUC.cv.tsv")
            cvauc.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符
    ## 
    ## -i -m
    elif args.model is not None: 
        model_dict= joblib.load(args.model)
        rf_model = model_dict['model']  # 获取模型
        feature_index=model_dict['feature_names']  # 获取模型特征
        ##重新排序,获得模型输入数据
        if args.table is not None:
                X=reorder_and_fill_features(args.table,X)
        else:
                X=reorder_and_fill_features(feature_index,X)	    
        df=predict_scores(rf_model,X,None,"predict")
        print(X.shape)
        # 保存为 tab 格式的文件
        output_file = os.path.join(args.output, f"{args.prefix}_.predict.scores.tsv")
        df.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符


if __name__ == "__main__":
    main()

