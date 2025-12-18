#!/usr/bin/env python

"""
RF
===============================================
Author: Chen Jing (cjwell@163.com)
"""
import io
import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from time import time
from sklearn.base import ClassifierMixin, RegressorMixin,clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict,learning_curve,StratifiedKFold,KFold,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc,f1_score,r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer  ##缺失值处理
from sklearn.inspection import permutation_importance
from time import time
import joblib

from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

##多方法学
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
##额外包
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

from scipy.cluster import hierarchy
from scipy.stats import spearmanr
import seaborn as sns

##重要性评估
import lime
import lime.lime_tabular
import shap


def analyze_correlations(X, y, A_filename="top_corr_data.txt", B_filename="R_and_pValue.txt", C_filename="corr.pdf",topnum=10):
    """
    分析X中各列与y的斯皮尔曼相关性，筛选top30显著变量，并输出相关结果和热图
    
    参数:
    X: 数据框，行是样本，列是特征
    y: Series，目标变量
    A_filename: 输出原始数据的文件名
    B_filename: 输出相关性结果的文件名
    C_filename: 输出热图的文件名
    """
    # 1. 计算X各列与y的斯皮尔曼相关性
    correlations = []
    p_values = []
    # 检测并过滤常数列
    non_constant_cols = []
    for col in X.columns:
        if np.std(X[col]) > 0:  # 标准差大于0表示不是常数
            non_constant_cols.append(col)

    # 只对非常数列计算相关系数
    X = X[non_constant_cols]

    for col in X.columns:
        r, p = spearmanr(X[col], y)
        correlations.append(r)
        p_values.append(p)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'feature': X.columns,
        'correlation': correlations,
        'p_value': p_values
    })
    
    # 2. 筛选显著特征（p < 0.05）
    significant_results = results_df[results_df['p_value'] < 0.05]
    
    # 如果没有显著特征，则退出
    if len(significant_results) == 0:
        print("没有发现显著相关的特征")
        return
    
    # 按p值排序并选择top30（或更少）
   # top_features = significant_results.sort_values('p_value').head(30)
    if topnum >10:
    # 按p值排序并选择前30个（不足30则取全部）
        #top_features = significant_results.sort_values('p_value').head(30)
        top_features = (significant_results
                .assign(corr_abs=significant_results['correlation'].abs())
                .sort_values(['p_value', 'corr_abs'], ascending=[True, False])
                .drop('corr_abs', axis=1)
                .head(30))
    elif topnum <0:  # balanced 模式
        topnum=abs(topnum)
        # 分离正负相关特征
        #pos = significant_results[significant_results['correlation'] > 0].sort_values('p_value')
        #neg = significant_results[significant_results['correlation'] < 0].sort_values('p_value')
        # 正相关：按p_value升序，再按correlation绝对值降序
        pos = (significant_results[significant_results['correlation'] > 0]
            .assign(corr_abs=lambda x: x['correlation'].abs())
            .sort_values(['p_value', 'corr_abs'], ascending=[True, False])
            .drop('corr_abs', axis=1))

        # 负相关：按p_value升序，再按correlation绝对值降序（注意负相关要取绝对值）
        neg = (significant_results[significant_results['correlation'] < 0]
            .assign(corr_abs=lambda x: x['correlation'].abs())
            .sort_values(['p_value', 'corr_abs'], ascending=[True, False])
            .drop('corr_abs', axis=1))
    
        # 计算每类要取的数量（最多10个）
        n_pos = min(topnum, len(pos))
        n_neg = min(topnum, len(neg))
    
        # 合并结果
        top_features = pd.concat([pos.head(n_pos), neg.head(n_neg)])
    print(top_features)

    # 3. 输出文件A：y和top特征的原始数据
    output_data = pd.concat([y, X[top_features['feature']]], axis=1)
    output_data.to_csv(A_filename, sep='\t', index=True)
    
    # 4. 计算top特征之间的相关性
    # 包括y和所有top特征
    all_features = [y.name] + list(top_features['feature'])
    correlation_matrix = pd.DataFrame(index=all_features, columns=all_features)
    pvalue_matrix = pd.DataFrame(index=all_features, columns=all_features)
    
    # 计算相关系数和p值
    for i, feat1 in enumerate(all_features):
        for j, feat2 in enumerate(all_features):
            if i <= j:  # 避免重复计算
                if feat1 == y.name:
                    data1 = y
                else:
                    data1 = X[feat1]
                
                if feat2 == y.name:
                    data2 = y
                else:
                    data2 = X[feat2]
                
                r, p = spearmanr(data1, data2)
                correlation_matrix.loc[feat1, feat2] = r
                correlation_matrix.loc[feat2, feat1] = r
                pvalue_matrix.loc[feat1, feat2] = p
                pvalue_matrix.loc[feat2, feat1] = p
    
    # 5. 输出文件B：相关性结果
    with open(B_filename, 'w') as f:
        f.write("Variable1\tVariable2\tR_value\tP_value\n")
        for i, feat1 in enumerate(all_features):
            for j, feat2 in enumerate(all_features):
                if i < j:  # 只输出上三角部分，避免重复
                    r = correlation_matrix.loc[feat1, feat2]
                    p = pvalue_matrix.loc[feat1, feat2]
                    f.write(f"{feat1}\t{feat2}\t{r}\t{p}\n")
    
    # 6. 绘制热图
    # 转换为数值矩阵
    corr_values = correlation_matrix.astype(float)
    p_values = pvalue_matrix.astype(float)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制热图
    sns.heatmap(corr_values, annot=False, fmt=".2f", cmap='RdBu_r', 
                center=0, square=True, ax=ax)
    
    # 添加显著性标记
    for i in range(len(corr_values)):
        for j in range(len(corr_values)):
            p_val = p_values.iloc[i, j]
            if p_val < 0.01:
                ax.text(j+0.5, i+0.5, "**", ha='center', va='center')
            elif p_val < 0.05:
                ax.text(j+0.5, i+0.5, "*", ha='center', va='center')
    
    # 设置标题和标签
    ax.set_title("Spearman Correlation Heatmap")
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(C_filename, format='pdf')
    plt.close()
    
    print(f"分析完成。结果已保存到 {A_filename}, {B_filename} 和 {C_filename}")


###进行causal_discovery因果推导
def run_causal_discovery(X: pd.DataFrame, y: pd.Series, output_file: str = "edges.txt", 
                         output_data: str = "node_data.txt", top_n_features: int = 30, 
                         alpha: float = 0.05, method: str = "PC"):
    """
    基于 PC 算法的因果结构发现流程，支持特征选择和标准化处理。
    
    参数:
    X -- 原始特征数据 (DataFrame)
    y -- 目标变量 (Series)
    output_file -- 保存边的文件名 (默认为 edges.txt)
    output_data -- 保存与target关联的数据文件名 (默认为 node_data.txt)
    top_n_features -- 基于随机森林的重要性选取的前 N 个特征 (默认 30)
    alpha -- PC 算法的显著性水平 (默认 0.05)
    method -- 因果发现方法 (默认 PC)
    """

    # 去除标准差过低的列
    X = X.loc[:, X.std() > 1e-8]

    if len(y.unique()) > 5:
        print("y超过5个unique值，将作为连续变量输入")
        # 使用随机森林回归器进行特征选择
        reg = RandomForestRegressor(random_state=42)
        reg.fit(X, y)
        importances = reg.feature_importances_
        indices = importances.argsort()[::-1][:top_n_features]
        X_selected = X.iloc[:, indices]
    else:
        print("y不超过5个unique值，将作为分类变量输入")
        # 使用随机森林进行降维
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_
        indices = importances.argsort()[::-1][:top_n_features]
        X_selected = X.iloc[:, indices] 

    # 将 y 合并到数据中
    # 使用 .loc 来明确指定赋值操作
    X_selected[y.name] = y.values

    # 标准化数据
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(X_selected), columns=X_selected.columns)
    df_scaled.index = X_selected.index  # 保留原始索引

    # 执行 PC 算法进行因果发现
    data = df_scaled.values

    if method == "PC":
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.GraphUtils import GraphUtils
        from causallearn.utils.cit import fisherz  # 可选：chisq, gsq, kci
        # 简洁处理：只替换有问题的字符

        clean_names = [
            str(name).replace(':', '_').replace('|', '_').replace(';', '_').strip() 
            for name in df_scaled.columns.tolist()
            ]
        print("使用 PC 算法进行因果发现...")
        data = df_scaled.values

        pc_graph = pc(data, alpha=0.05, ci_test=fisherz, node_names=clean_names)

        # 先尝试显示图形
        try:
            pc_graph.draw_pydot_graph()
        except:
            print("图形显示失败，继续保存文件...")

        # 保存为 PDF 文件
        save_pc_graph(pc_graph, "causal_graph.pdf")

        # 输出边信息到文件
        with open(output_file, "w") as f:
            for edge in pc_graph.G.get_graph_edges():
                f.write(str(edge) + "\n")
        print(f"因果边关系已保存到 {output_file}")
        
        # 提取与target节点相连的所有节点
        target_related_nodes = extract_target_related_nodes(pc_graph, df_scaled.columns.tolist())
        
        # 保存与target相关的数据
        save_target_related_data(X_selected, target_related_nodes, output_data)


def extract_target_related_nodes(pc_graph, all_nodes):
    """
    提取与'target'节点相连的所有节点
    
    参数:
    pc_graph -- PC算法得到的图
    all_nodes -- 所有节点的名称列表
    
    返回:
    与'target'节点相连的节点名称列表
    """
    target_related_nodes = set(['target'])
    
    # 获取所有边
    edges = pc_graph.G.get_graph_edges()
    
    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        
        node1_name = node1.get_name()
        node2_name = node2.get_name()
        
        # 如果任意一端是'target'，则将另一端加入相关节点集合
        if node1_name == 'target':
            target_related_nodes.add(node2_name)
        elif node2_name == 'target':
            target_related_nodes.add(node1_name)
    
    # 确保返回的节点列表按照原始顺序
    result = [node for node in all_nodes if node in target_related_nodes]
    return result


def save_target_related_data(df_scaled, target_related_nodes, output_file):
    """
    保存与target相关的数据到文件
    
    参数:
    df_scaled -- 标准化后的数据DataFrame
    target_related_nodes -- 与target相关的节点名称列表
    output_file -- 输出文件名
    """
    # 提取相关列的数据
    related_data = df_scaled[target_related_nodes]
    
    # 保存数据到文件
    with open(output_file, 'w') as f:
        # 写入列名
        f.write("ID\t"+"\t".join(related_data.columns) + "\n")
        
        # 写入每一行数据
        for idx, row in related_data.iterrows():
            # 写入行名和对应的数据
            f.write(f"{idx}\t" + "\t".join([str(x) for x in row]) + "\n")
    
    print(f"与target相关的数据已保存到 {output_file}")
    print(f"包含的节点: {', '.join(target_related_nodes)}")


def save_pc_graph(pc_graph, filename="causal_graph.pdf"):
    import pydot
    
    # 使用横向布局
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')
    
    node_names = pc_graph.G.get_node_names()
    
    # 添加节点
    for name in node_names:
        graph.add_node(pydot.Node(name, label=name))
    
    # 添加边
    edges = pc_graph.G.get_graph_edges()
    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        
        src = node1.get_name()
        dst = node2.get_name()
        
        # 获取端点类型，确保它们是字符串
        endpoint1 = str(getattr(edge, 'endpoint1', None))
        endpoint2 = str(getattr(edge, 'endpoint2', None))
        
        # 创建边对象
        pydot_edge = pydot.Edge(src, dst)
        
        # 根据端点类型设置箭头样式
        if endpoint1 == 'TAIL' and endpoint2 == 'ARROW':
            pydot_edge.set_arrowhead('normal')
            pydot_edge.set_arrowtail('none')
            pydot_edge.set_dir('forward')  # 明确设置为有向边
        elif endpoint1 == 'ARROW' and endpoint2 == 'TAIL':
            pydot_edge.set_arrowhead('none')
            pydot_edge.set_arrowtail('normal')
            pydot_edge.set_dir('back')  # 反向边
        elif endpoint1 == 'TAIL' and endpoint2 == 'TAIL':
            pydot_edge.set_arrowhead('none')
            pydot_edge.set_arrowtail('none')
            pydot_edge.set_dir('none')  # 无向边
        elif endpoint1 == 'ARROW' and endpoint2 == 'ARROW':
            pydot_edge.set_arrowhead('normal')
            pydot_edge.set_arrowtail('normal')
            pydot_edge.set_dir('both')  # 双向边
        else:
            # 未知结构作为无向边处理
            pydot_edge.set_arrowhead('none')
            pydot_edge.set_arrowtail('none')
            pydot_edge.set_dir('none')
        
        graph.add_edge(pydot_edge)
    
    # 尝试多种保存方式
    try:
        # 首先尝试PDF格式
        graph.write_pdf(filename)
    except Exception as e:
        print(f"无法保存为PDF: {e}")
        try:
            # 如果PDF失败，尝试PNG格式
            if filename.endswith('.pdf'):
                filename = filename.replace('.pdf', '.png')
            graph.write_png(filename)
            print(f"已保存为PNG格式: {filename}")
        except Exception as e2:
            print(f"也无法保存为PNG: {e2}")
            # 最后尝试DOT格式
            if filename.endswith('.pdf') or filename.endswith('.png'):
                filename = filename.replace('.pdf', '.dot').replace('.png', '.dot')
            graph.write_dot(filename)
            print(f"已保存为DOT格式: {filename}")


##获得初始模型和超参数网格
def get_model_and_params(method='random_forest', random_state=42):
    """
    根据指定的方法返回对应的机器学习模型和超参数网格
    
    参数:
    method -- 模型方法名称 (default 'random_forest')
    random_state -- 随机种子 (default 42)
    
    返回:
    tuple: (模型对象, 超参数网格字典)
    
    支持的 method:
    - 'random_forest': 随机森林
    - 'decision_tree': 决策树
    - 'logistic_regression': 逻辑回归
    - 'svm': 支持向量机
    - 'mlp': 多层感知机
    - 'knn': K近邻
    - 'naive_bayes': 朴素贝叶斯
    - 'gradient_boosting': 梯度提升
    - 'adaboost': AdaBoost
    - 'lda': 线性判别分析  
    """
    
    # 定义各模型的超参数网格
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        },
        'decision_tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2', None]
        },
        'logistic_regression': {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 500]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto'] + [0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4]  # 仅对poly核有效
        },
        'mlp': {
            'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [200, 500]
        },
        'knn': {
            'n_neighbors': list(range(1, 31)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]  # 1:曼哈顿距离, 2:欧式距离
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        },
        'adaboost': {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.5, 0.75, 1.0, 1.25, 1.5],
            'algorithm': ['SAMME', 'SAMME.R']
        },
        'lasso': {  # >>> 新增超参数网格
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
            'max_iter': [1000, 5000, 10000],
            'tol': [1e-4, 1e-5],
            'selection': ['cyclic', 'random']
        }
    }
    
    # 返回模型和对应参数网格
    if method == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
        return model, param_grids['random_forest']
    elif method == 'decision_tree':
        model = DecisionTreeClassifier(random_state=random_state)
        return model, param_grids['decision_tree']
    elif method == 'logistic_regression':
        model = LogisticRegression(random_state=random_state)
        return model, param_grids['logistic_regression']
    elif method == 'svm':
        model = SVC(probability=True, random_state=random_state)
        return model, param_grids['svm']
    elif method == 'mlp':
        model = MLPClassifier(random_state=random_state)
        return model, param_grids['mlp']
    elif method == 'knn':
        model = KNeighborsClassifier()
        return model, param_grids['knn']
    elif method == 'naive_bayes':
        model = GaussianNB()
        return model, {}  # 朴素贝叶斯通常不需要调参
    elif method == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=random_state)
        return model, param_grids['gradient_boosting']
    elif method == 'adaboost':
        model = AdaBoostClassifier(random_state=random_state)
        return model, param_grids['adaboost']
    elif method == 'lda':
        model = LinearDiscriminantAnalysis()
        return model, {}  # LDA通常不需要调参
    elif method == 'lasso':  # >>> 新增判断逻辑
        model = Lasso(random_state=random_state, max_iter=10000)
        return model, param_grids['lasso']
    else:
        raise ValueError(f"Unknown method: {method}")

###超参数优化

def optimize_model(method='random_forest', X=None, y=None, feature_names=None,
                  random_state=42, cv=5, n_iter=50, search_method='random',
                  output_dir='./',prefix='md',scoring=None):
    """
    自动优化指定类型的模型并输出特征重要性（如果可用）
    
    新增参数:
    feature_names -- 特征名称列表
    output_dir -- 结果保存目录
    """

    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取基础模型和参数网格
    model, param_grid = get_model_and_params(method, random_state)
    
    if not param_grid:
        print(f"{method} 模型没有可调参数，返回基础模型")
        if X is not None and y is not None:
            model.fit(X, y)
            #_save_feature_importance(model, method, X, y, feature_names, output_dir)
            return model
        return model
    
    # 选择搜索方法
    if search_method == 'grid':
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
    elif search_method == 'random':
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
    else:
        raise ValueError("search_method 必须是 'grid' 或 'random'")
    
    if X is not None and y is not None:
        start_time = time()
        search.fit(X, y)
        print(f"训练耗时: {time()-start_time:.2f}秒")
        print(f"最佳参数: {search.best_params_}")
        print(f"最佳分数: {search.best_score_:.4f}")

        #保存最佳参数
        best_params=search.best_estimator_.get_params()
        output_file = os.path.join(output_dir, f"{prefix}_best.params.txt")
        # 打开一个txt文件并保存字典
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"最佳分数: {search.best_score_:.4f}\n") 
            for key, value in search.best_params_.items():
                f.write(f"{key}: {value}\n")  # 每对键值对后面加一个换行符       
        # 保存特征重要性
        #_save_feature_importance(search.best_estimator_, method, X, y, feature_names, output_dir,prefix=prefix,scoring=scoring)
        # 保存模型
        _save_model_pkl(search.best_estimator_, method, output_dir,prefix=prefix,X=X,y=y)
   
        return search.best_estimator_
    else:
        return model



def _save_feature_importance(model, method, X, y, feature_names=None,
                             output_dir='./', prefix='md', top_n_features=30,scoring=None):
    """
    优化版特征重要性保存函数：
    - 支持分类或连续y变量的特征富集/相关性分析；
    - 输出 top_n_features 的 X 子集；
    """

    os.makedirs(output_dir, exist_ok=True)

    # 初始化特征名
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # 转 DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    # 收集特征重要性
    results = {'feature': feature_names}
    importance_scores = None
    start_time = time()

    # 内置特征重要性
    if method in ['random_forest', 'decision_tree', 'gradient_boosting', 'adaboost']:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            results['importance'] = importances
            importance_scores = importances
            print(f"\n模型内置特征重要性计算完毕 (耗时 {time()-start_time:.2f}s)")

    # 排列重要性（其他模型）
    if importance_scores is None:
        print("\n计算排列重要性...")
        perm_start = time()
        perm = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        results['importance'] = perm.importances_mean
        results['importance_std'] = perm.importances_std
        importance_scores = perm.importances_mean
        print(f"排列重要性计算耗时: {time()-perm_start:.2f}s")

    # 构建 DataFrame
    df = pd.DataFrame(results)
    if 'importance_std' not in df:
        df['importance_std'] = np.nan
    df['importance_rank'] = df['importance'].rank(ascending=False)
    df = df.sort_values('importance', ascending=False)

    # 仅取前 top_n_features
    df_top = df.head(top_n_features).copy()

    # 判断 y 类型：分类 or 连续
    y_series = pd.Series(y)
    is_classification = (y_series.dtype == 'object' or y_series.dtype.name == 'category' or
                         (y_series.nunique() < 10 and not np.issubdtype(y_series.dtype, np.number)))

    # 分类情况：计算富集分组和均值
    if is_classification:
        print("\n检测到 y 为分类变量，计算富集分组与均值...")
        enrich_list, mean_list = [], []

        for feat in df_top['feature']:
            means = X.groupby(y_series)[feat].mean()
            enrich_group = means.idxmax()
            enrich_list.append(enrich_group)
            mean_list.append(means.loc[enrich_group])

        df_top['Enrich'] = enrich_list
        df_top['Mean'] = mean_list

    # 连续情况：计算相关性
    else:
        print("\n检测到 y 为连续变量，计算 Spearman 相关性...")
        cors, pvals = [], []
        for feat in df_top['feature']:
            cor, pval = spearmanr(X[feat], y_series)
            cors.append(cor)
            pvals.append(pval)
        df_top['Cor'] = cors
        df_top['Pvalue'] = pvals

    # 输出文件
    output_path = os.path.join(output_dir, f"{prefix}_importances.tsv")
    df_top.to_csv(output_path, sep='\t', index=False)
    print(f"\n前 {top_n_features} 个特征重要性结果已保存至: {output_path}")

    # 保存前 top_n_features 的数据（含 y）
    X_top = X[df_top['feature']].copy()
    X_top.insert(0, 'target', y_series.values)
    top_features_path = os.path.join(output_dir, f"{prefix}_top{top_n_features}_features.tsv")
    X_top.to_csv(top_features_path, sep='\t', index=True)
    print(f"前 {top_n_features} 特征数据（含 y）已保存至: {top_features_path}")



    # 10. 计算基于组合性能的重新排序
    if scoring is not None:
        selected_features, score_record = reorder_top_features(X_top, target_col='target', model=model, scoring=scoring)
        
        # 保存为新文件
        score_path = os.path.join(output_dir, f"{prefix}_top{top_n_features}_score.tsv")
        with open(score_path, 'w') as f:
            f.write("Step\tFeatures\tScore\n")
            for step, (feat_list, score) in enumerate(score_record, 1):
                f.write(f"{step}\t{','.join(feat_list)}\t{score:.4f}\n")
        print(f"前{top_n_features}特征基于 {scoring} 的组合排序已保存至: {score_path}")


def reorder_top_features(X_top, target_col='target', model=None, scoring='auc'):
    """
    根据组合得分重新排序Top特征
    
    参数：
        X_top: DataFrame, 第一列是target，其余列是特征
        target_col: y列名称
        model: 可选模型，分类或回归
        scoring: 评价指标，支持 'auc', 'accuracy', 'f1', 'r2', 'mse', 'mae'
    返回：
        selected_features: 根据得分重新排序的特征列表
        score_record: 每步组合对应的得分
    """
    if model is None:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='liblinear')
    else:
        model=clone(model)
    y = X_top[target_col].values
    features = [col for col in X_top.columns if col != target_col]
    
    selected = []
    remaining = features.copy()
    score_record = []

    def evaluate(y_true, y_pred, y_prob=None):
        if scoring == 'auc':
            if y_prob is None:
                raise ValueError("AUC需要概率输出")
            return roc_auc_score(y_true, y_prob)
        elif scoring == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif scoring == 'f1':
            return f1_score(y_true, y_pred)
        elif scoring == 'r2':
            return r2_score(y_true, y_pred)
        elif scoring == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif scoring == 'mae':
            return mean_absolute_error(y_true, y_pred)
        else:
            raise ValueError(f"未知的scoring方法: {scoring}")



    # 遍历 1 到 n 个特征
    for i in range(1, len(remaining) + 1):
        current_features = remaining[:i]
        model.fit(X_top[current_features], y)
        y_pred = model.predict(X_top[current_features])
        y_prob = model.predict_proba(X_top[current_features])[:, 1] if hasattr(model, "predict_proba") else None
        print(y_pred)
        score = evaluate(y, y_pred, y_prob)
        print(score)
        score_record.append((current_features.copy(), score))

    """
    # 第一步：选择单个特征表现最好的
    best_score = -1e9
    for f in remaining:
        model.fit(X_top[[f]], y)
        y_pred = model.predict(X_top[[f]])
        y_prob = model.predict_proba(X_top[[f]])[:, 1] if hasattr(model, "predict_proba") else None
        score = evaluate(y, y_pred, y_prob)
        if score > best_score:
            best_score = score
            best_feature = f
    selected.append(best_feature)
    remaining.remove(best_feature)
    score_record.append((selected.copy(), best_score))

    # 迭代选择剩余特征
    while remaining:
        best_score = -1e9
        for f in remaining:
            model.fit(X_top[selected + [f]], y)
            y_pred = model.predict(X_top[selected + [f]])
            y_prob = model.predict_proba(X_top[selected + [f]])[:, 1] if hasattr(model, "predict_proba") else None
            score = evaluate(y, y_pred, y_prob)
            if score > best_score:
                best_score = score
                best_feature = f
        selected.append(best_feature)
        remaining.remove(best_feature)
        score_record.append((selected.copy(), best_score))
     """
	   
    return selected, score_record

def _save_model_pkl(model, method, output_dir='./', prefix='md', overwrite=False,X=None,y=None):
    """将模型保存为pkl文件
    
    参数:
        model: 训练好的模型
        method: 模型方法名称
        output_dir: 输出目录 (默认当前目录)
        prefix: 文件名前缀 (默认'md')
        overwrite: 是否覆盖已有文件 (默认False)
    """
    os.makedirs(output_dir, exist_ok=True)
    model_file = os.path.join(output_dir, f"{prefix}_model.pkl")
    
    if os.path.exists(model_file) and not overwrite:
        print(f"警告：模型文件 {model_file} 已存在，跳过保存")
        return model_file
    """
    # 保存模型和特征信息
    save_data = {
        'model': model,
        'feature_names': getattr(model, 'feature_names_in_', None),
        'method': method
    }
    joblib.dump(save_data, model_file)
    """
###和X，y一起保存
    joblib.dump({
    'model': model,
    'feature_names': getattr(model, 'feature_names_in_', None),
    'x': X,
    'y': y,
    'method': method
    }, model_file)
    print(f"模型已保存至: {model_file}")
    return model_file


def onehot_auto(X):
    """
    对 DataFrame X 的每一列执行以下操作：
    - 如果该列（包括空值）是可转换为数值的：将空值填充为0；
    - 如果该列包含非数值元素：进行 one-hot 编码（包括 NaN -> 'NA'），结果仍为纯数值；
    
    返回处理后的 DataFrame，所有列均为数值类型。
    """
    processed_cols = []

    for col in X.columns:
        # 判断该列是否为纯数值列
        #print(f"检查列: {col}, 类型: {type(X[col])}, shape: {X[col].shape}")
        try:
            pd.to_numeric(X[col].dropna(), errors='raise')
            # 是数值列
            processed_col = pd.to_numeric(X[col], errors='coerce').fillna(0)
            processed_cols.append(processed_col.rename(col))
        except ValueError:
            # 是非数值列：进行 one-hot 编码，包括空值视作 'NA'
            onehot = pd.get_dummies(X[col].fillna('NA'), prefix=col, dtype=int)
            processed_cols.append(onehot)

    # 合并所有列
    result = pd.concat(processed_cols, axis=1)
    return result

##bar图
def plot_top_bars(df, col1, col2, col3, top_n, colors,output_file):
    # 确保传入的列名存在于数据框中
    if col1 not in df.columns or col2 not in df.columns or col3 not in df.columns:
        raise ValueError("指定的列名在数据框中不存在")

    # 根据列2的值从高到低排序，并选择前top_n个
    sorted_df = df.nlargest(top_n, col2)

    # 获取唯一分类变量和颜色映射
    unique_categories = sorted_df[col3].unique()
    color_map = {cat: colors[i] if i < len(colors) else 'grey' for i, cat in enumerate(sorted(unique_categories))}

    # 如果分类变量数量超过颜色队列长度，使用灰色
    default_color = 'gray'
    sorted_df['color'] = sorted_df[col3].map(color_map).fillna(default_color)
    print(sorted_df)
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


def predict_scores(model, X, y=None, run="run"):
    if isinstance(model, ClassifierMixin):
        # 分类模型
        y_proba = model.predict_proba(X)
        y_pred = model.predict(X)
        results = pd.DataFrame(y_proba, columns=model.classes_)
        results['sample_name'] = X.index
        results['predicted_class'] = y_pred
        results['run'] = run
        if y is not None:
            results['actual_class'] = y.tolist()
            results = results[['sample_name'] + list(model.classes_) + ['predicted_class', 'actual_class', 'run']]
        else:
            results = results[['sample_name'] + list(model.classes_) + ['predicted_class', 'run']]
    elif isinstance(model, RegressorMixin):
        # 回归模型
        y_pred = model.predict(X)
        results = pd.DataFrame({
            'sample_name': X.index,
            'predicted_value': y_pred,
            'run': run
        })
        if y is not None:
            results['actual_value'] = y.tolist()
            results = results[['sample_name', 'predicted_value', 'actual_value', 'run']]
        else:
            results = results[['sample_name', 'predicted_value', 'run']]
    else:
        raise ValueError("模型既不是分类器也不是回归器，无法处理。")

    return results



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
        features=index.tolist()
    
    
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
    parser = argparse.ArgumentParser(description="sklearn Model using -i for feature table and -g for binary phenotype grouping.\n使用-m 载入")
    parser.add_argument("-i", "--feature_table", required=True, help="输入特征队列，column为feature，row为样本")
    parser.add_argument("-it", "--feature_transpose", action='store_true', help="当-i column为样本，row为feature时，转置特征队列")
    parser.add_argument("-g", "--phenotype", required=False, help="表型分组，第一列样本，第二列表型分组")
    parser.add_argument("-gc", "--group_column", required=False , type=int,default=1,help="表型分组的第几列用于模型")
    parser.add_argument("-gp", "--group_pick", required=False , nargs="+",default=None,help="从group中选择特定的分组用于分析格式：列号+关键词，默认关闭") 
    parser.add_argument("-m", "--model", required=False, help="输入model进行预测 file.pkl")
    parser.add_argument("-o", "--output", required=False, default="./" , help="Output directory")
    parser.add_argument("-p", "--prefix", required=False, default=None , help="指定文件抬头")
    parser.add_argument("-mtd", "--method", required=False, default="RF",choices=["RF","DT","LGR","SVM","MLP","KNN","NBS","GBS","ABS","LDA","lasso"],help="模型预测，选择方法默认RF（随机森林），RF/DT/LGR/SVM/MLP/KNN/NBS/GBS/ABS/LDA/lasso")
    parser.add_argument("-mn", "--minnum", required=False, type=int,default=50 , help="构建模型时每个分类的最小样本数")
    parser.add_argument("-s", "--scoring", required=False, choices=["auc","accuracy","f1","r2","mse","mae"],default=None , help="进行评价的指标,回归评价使用r2 mae mse")
    parser.add_argument("-ta", "--target", required=False , default=None,help="指定绘制ROC曲线中预测的对象")
    parser.add_argument("-tb", "--table", required=False , help="如果-m模型中不含feature name，输入-tb 使用imortances.tsv提取feature name")
    parser.add_argument("-opt", "--optimize", required=False ,default="random", help="超参数选择,可选grid，random（默认）和None(不进行超参数优化，不生成pkl文件)")
    parser.add_argument("-cc", "--cf_counts", required=False , type=int,default=100,help="学习曲线和AUC.cv评价的最低样本数，过低的数量可能导致抽样问题")
    parser.add_argument("-mtdC", "--method_Case", required=False, default=None ,choices=["PC"], help="因果推断causallearn包，适用较大样本量，目前仅支持PC（需要n > 5 × 特征数），PC/FCI/RFCI/GES/LINGAM/NOTEARS")
    parser.add_argument("-C1", "--Case_pam1", required=False,type=int,default=30 ,  help="因果推断topfeature数")       
    parser.add_argument("-mtdCor", "--method_Cor", required=False, default=None ,choices=["SP"], help="相关性分析")
    parser.add_argument("-Cor1", "--Cor_pam1", required=False,type=int,default=30 , help="相关性分析topfeature数")          
    args = parser.parse_args()
    # 创建输出目录（如果不存在）
    os.makedirs(args.output, exist_ok=True)

###指定方法
    if args.method == 'RF':
        method = 'random_forest'
    elif args.method == 'DT':
        method = 'decision_tree'
    elif args.method == 'LGR':   ##仅分类
        method = 'logistic_regression'
    elif args.method == 'SVM':
        method = 'svm'
    elif args.method == 'MLP':
        method = 'mlp'
    elif args.method == 'KNN':
        method = 'knn'
    elif args.method == 'NBS':   ##仅分类
        method = 'naive_bayes'
    elif args.method == 'GBS':
        method = 'gradient_boosting'
    elif args.method == 'ABS':
        method = 'adaboost'
    elif args.method == 'LDA':   ##仅分类
        method = 'lda'
    elif args.method == 'lasso':  #仅回归
        method = 'lasso'
    elif args.method == 'None':
        method = None
    else:
        print("请选择正确的-mtd 参数")
        sys.exit()

    if args.prefix is not None:
        args.prefix=args.prefix
    elif args.model is not None:
        args.prefix="model"
    elif  args.prefix is None or args.prefix =="":
        args.prefix=args.method
    else:
        args.prefix=f"{args.prefix}_{args.method}"

    ###-i
    #X = pd.read_csv(args.feature_table, sep='\t', index_col=0, dtype={'column_name': float},low_memory=False)
    X = pd.read_csv(args.feature_table, sep='\t', index_col=0, dtype=str, low_memory=False)
    if args.feature_transpose:
	    X=X.T
	    print("使用-it参数转置X输入，行名为sample，列名为特征")
	    #print(X)
    
    #-i -g 输入
    if args.phenotype is not None:
        # 读取表型分组信息文件
        y = pd.read_csv(args.phenotype, sep='\t', index_col=0,dtype={'column_name': str},low_memory=False)

        if args.group_pick is not None:
            # 提取完整行子集,格式列+关键词
            if len(args.group_pick)>=2:
                pcn=int(float(args.group_pick[0]))
                pcn=pcn-1
                pick=args.group_pick[1:]
                print(f"提取 {y.columns[pcn]} 中 f{pick} 子集")
                y= y[y.iloc[:, pcn].isin(pick)]
        gc=args.group_column
        gc=gc-1

	# 忽略第二列名字为空的行
        y = y.dropna(subset=[y.columns[gc]])
        # 合并两个数据框，只保留共有的样本
        print(f"选择 {y.columns[gc]} 作为主变量进行分析")
        y = y.iloc[:,gc] ##选择数据


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

        if args.model is None and merged_data.shape[0] < args.minnum:
            print(f"样本数不足{args.minnum}（当前最小样本数为 {merged_data.shape[0]}），跳过此任务，如需继续请修改-mn参数")
            sys.exit(0)  # 正常退出


        ##检验X是否存在分类变量
        X=onehot_auto(X)

        ##检测-ta值
        unique_y=y.unique()
        if args.target is not None and args.target not in unique_y:
            print(f" -ta 取值 为 {unique_y[0]}")
            args.target=unique_y[0]
        if len(unique_y)==1:
            print(f"y取值只有1个，退出")
            sys.exit()
        
	#
        #
	##执行因果推断
        if args.method_Case is not None:
            if args.method_Case == "PC":
                run_causal_discovery(X,y,top_n_features=args.Case_pam1)
            sys.exit(0)  #关闭模型，正常退出
	##执行相关性分析
        if args.method_Cor is not None:
            if args.method_Cor == "SP":
                analyze_correlations(X,y,topnum=args.Cor_pam1)
            sys.exit(0)  #关闭模型，正常退出	
	###
        #
        #

        if method is None:
            sys.exit(0)  #关闭模型，正常退出

	# 初始模型,输入值有模型时进行下一步
        int_model, param_grid = get_model_and_params(method=method)

	#预设model  -i -g -m
        if args.model is not None:
            model_dict= joblib.load(args.model)
            vd_model = model_dict['model']  # 获取模型
            X_train=model_dict['x']  # 获取模型x
            y_train=model_dict['y']  # 获取模型y
            X_test=X  # 获取输入x
            y_test=y  # 获取输入y
            ##重新排序特征
            feature_index=model_dict['feature_names']  # 获取模型特征
            ##重新排序,获得模型输入数据
            if args.table is not None:
                X_test=reorder_and_fill_features(args.table,X_test)
            else:
                X_test=reorder_and_fill_features(feature_index,X_test)

            df1=predict_scores(vd_model,X_train,y_train,"train")
            df2=predict_scores(vd_model,X_test,y_test,"validation")

            df = pd.concat([df1, df2], axis=0)
            # 保存为 tab 格式的文件
            output_file = os.path.join(args.output, f"{args.prefix}_validation.scores.tsv")
            df.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符

	    ##绘制ROC曲线
            y_groups=y_train.unique()
            if len(y_groups) >5: ##超过5分类不执行
                print("-g 超过5个分类,跳过ROC绘制")
                #sys.exit()
            elif len(y_groups) ==2: ##二分类
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

	##重新计算 -i -g 继续
        else:
            #"""
            ##记录分布
            distribution = y.value_counts()
            output_file = os.path.join(args.output, f"{args.prefix}_train.distribution.txt")
            distribution.to_csv(output_file, sep='\t', header=True)    
    
	    ##学习曲线评价
            y_groups=y.unique() ##分类数目
            if len(y_groups) >5: ##超过5分类不执行
                if method in ["lasso"]:
                    print(f"-g 超过5个分类,-mtd参数为{args.method},-g输入连续变量值跳过绘制学习曲线")
                else:
                    print(f"-g 超过5个分类,-mtd参数为{args.method},-g输入分类变量过多不能绘制学习曲线，退出！")
                    #sys.exit()
            elif len(y_groups) ==2:  ##二分类评价
                print("进行二分类评价")
                ty=y.copy()
                # 绘制学习曲线
                output_pdf=os.path.join(args.output, f"{args.prefix}_LearningCurves.pdf")
                if os.path.exists(output_pdf):
                    print(f"警告：输出文件 {output_pdf} 已存在，跳过!")
                else:
                    plot_learning_curve(int_model,X,ty,output_pdf,"accuracy")

            elif len(y_groups) >2:   ##3-5类评价
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
                    if os.path.exists(output_pdf):
                        print(f"警告：输出文件 {output_pdf} 已存在，跳过!")
                    else:
                        plot_learning_curve(int_model,X,ty,output_pdf,score,title) ##模型，x，y，输出文件，评分方法(roc_auc_ovr,roc_auc,accuracy)，标题
###

            ##绘制cv ROC曲线
            if len(y_groups) >5: ##超过5分类不执行
                print(f"-g 超过5个分类,-mtd参数为{args.method},-g需要输入分类变量,连续变量值不能绘cv ROC")
            else:
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
                    output_file=os.path.join(args.output, f"{args.prefix}_test.{y_target}.roc.cv.pdf")
                    if os.path.exists(output_file):
                        print(f"警告：输出文件 {output_file} 已存在，跳过!")
                    else:
                        auc_tmp=plot_cv_roc(int_model,X,ty,5,y_target,output_file)
                        print(auc_tmp)
                        ##记录每个分类的AUC
                        auc_list.append(auc_tmp)   

                # 保存全部过程AUC
                output_file = os.path.join(args.output, f"{args.prefix}_test.AUC.cv.tsv")
                if os.path.exists(output_file):
                    print(f"警告：输出文件 {output_file} 已存在，跳过!")
                else:
                    if auc_list:  # 检查auc_list是否非空
                        cvauc = pd.concat(auc_list, ignore_index=True)  # 垂直合并
                        cvauc.to_csv(output_file, sep='\t', index=False)
                        print(f"结果已保存至: {output_file}")
                    else:
                        print("警告：auc_list为空，没有数据需要保存！")

###################超参数优化
            model_file = os.path.join(args.output, f"{args.prefix}_model.pkl")

            if args.optimize == "None":
                print(f"-opt参数使用了None，跳过超参数调优模型，未生成pkl模型文件!")
            elif os.path.exists(model_file):
                print(f"警告：模型输出文件 {output_file} 已存在，跳过超参数优化!")
            else:
                # 优化模型并保存模型及特征重要性
                print("正在进行超参数优化")
                feature_names = X.columns.tolist() 
                optimized_md = optimize_model(
                    method=method,
                    X=X,
                    y=y,
                    feature_names=feature_names,
                    search_method=args.optimize,
                    output_dir=args.output,
		    prefix=args.prefix,
		    scoring=args.scoring
                )
                # 保存特征重要性
                _save_feature_importance(optimized_md, method, X, y, feature_names, output_dir=args.output,prefix=args.prefix,scoring=args.scoring)
                ##保存预测分
                df=predict_scores(optimized_md,X,y,"train")

                # 保存为 tab 格式的文件
                output_file = os.path.join(args.output, f"{args.prefix}_train.scores.tsv")
                df.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符
                #output_file = os.path.join(args.output, f"{args.prefix}_train.roc.pdf")
                #auc=df_roc_curve(df,"actual_class","actual_class",args.target,output_file) ## df，subset,actual,target,输出pdf路径
                #pd.DataFrame(auc)
                # 保存AUC
                #output_file = os.path.join(args.output, f"{args.prefix}_train.AUC.tsv")
                #auc.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符  

            #"""
    
    ## 
    ## -i -m
    elif args.model is not None: 
        model_dict= joblib.load(args.model)
        pre_model = model_dict['model']  # 获取模型
        feature_index=model_dict['feature_names']  # 获取模型特征
        ##重新排序,获得模型输入数据
        if args.table is not None:
                X=reorder_and_fill_features(args.table,X)
        else:
                X=reorder_and_fill_features(feature_index,X)	    
        df=predict_scores(pre_model,X,None,"predict")
        print(X.shape)

        # 保存为 tab 格式的文件
        output_file = os.path.join(args.output, f"{args.prefix}_predict.scores.tsv")
        df.to_csv(output_file, sep='\t', index=False)  # 使用制表符作为分隔符


if __name__ == "__main__":
    main()

