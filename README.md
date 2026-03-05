<h1><mark><strong>ModelFlow: Machine Learning Modeling Pipeline  / 机器学习建模流程</strong></mark></h1>  
English | 中文

<h2><mark><strong>English</strong></mark></h2>    
<h2><mark><strong>📖 Introduction</strong></mark></h2>    
ModelFlow is a comprehensive machine learning tool based on scikit-learn, supporting various classification and regression algorithms. It provides end-to-end functionality from data preprocessing, model training, hyperparameter optimization to result visualization and causal inference, making machine learning workflow smooth and efficient.  

<h2><mark><strong>✨ Features</strong></mark></h2>  
Multiple Algorithms: Random Forest, Decision Tree, Logistic Regression, SVM, MLP, KNN, Naive Bayes, Gradient Boosting, AdaBoost, LDA, Lasso<br>	  
Data Preprocessing: Automatic one-hot encoding, missing value handling, feature correlation analysis<br>
Model Optimization: Grid search and random search for hyperparameter optimization<br>	  
Feature Importance: Built-in importance and permutation importance analysis<br>
Model Evaluation: ROC curves, learning curves, cross-validation, various metrics (AUC, accuracy, F1, R2, MSE, MAE)<br>  
Visualization: Correlation heatmaps, ROC curves, learning curves, feature importance plots<br>  
Causal Inference: PC algorithm-based causal discovery (requires causallearn package)<br> 
Model Persistence: Save/load models using joblib<br> 
Prediction: Apply trained models to new data <br>

<h2><mark><strong>🛠️ Installation</strong></mark></h2>
Dependencies<br>  
pip install pandas matplotlib numpy scikit-learn scipy seaborn joblib lime shap<br>  
<br>  
Optional dependencies (for causal inference)<br>  
pip install causallearn pydot<br>

<h2><mark><strong>🚀 Usage</strong></mark></h2>
Basic Command Structure <br> 
python ModelFlow.py -i feature_table.txt -g phenotype.txt [options]<br>

Required Arguments  <br>
-i, --feature_table: Input feature table (rows = samples, columns = features)  <br>
-g, --phenotype: Phenotype grouping file (first column = sample IDs, second column = groups)  <br>
<br>
Optional Arguments  <br>
Argument	Description	Default  
-it, --feature_transpose	Transpose feature table if samples are columns	False  <br>
-gc, --group_column	Which column in phenotype file to use (1-based) 1   	<br>
-gp, --group_pick	Select specific groups (format: column+keywords)	None<br>
-m, --model	Load existing model for prediction	None<br>
-o, --output	Output directory	./<br>
-p, --prefix	Output file prefix	method name<br>
-mtd, --method	Algorithm (RF/DT/LGR/SVM/MLP/KNN/NBS/GBS/ABS/LDA/lasso)	RF<br>
-mn, --minnum	Minimum samples per class	50<br>
-s, --scoring	Evaluation metric	None<br>
-ta, --target	Target class for ROC curves	None<br>
-tb, --table	Feature importance file for model prediction	None<br>
-opt, --optimize	Hyperparameter search (grid/random/None)	random<br>
-cc, --cf_counts	Minimum samples for learning curves	100<br>
-mtdC, --method_Case	Causal inference method (PC)	None<br>
-C1, --Case_pam1	Number of top features for causal inference	30<br>
-mtdCor, --method_Cor	Correlation analysis (SP)	None<br>
-Cor1, --Cor_pam1	Number of top features for correlation	30<br>
Examples<br>
Train a Random Forest model:<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtd RF -o ./results -p my_project<br>
<br>
Hyperparameter optimization with grid search:<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtd SVM -opt grid -o ./results<br>
<br>
Apply existing model to new data:<br>
python ModelFlow.py -i new_features.tsv -m trained_model.pkl -tb feature_importance.tsv -o ./predictions<br>
<br>
Perform correlation analysis:<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtdCor SP -Cor1 20 -o ./correlation<br>
<br>
Causal inference (requires causallearn):<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtdC PC -C1 30 -o ./causal<br>

<h2><mark><strong>📁 Output Files</strong></mark></h2>  
Model Training<br>
{prefix}_model.pkl: Trained model (with features and training data)<br>
{prefix}_best.params.txt: Best hyperparameters<br>
{prefix}_importances.tsv: Feature importance scores<br>
{prefix}_top{num}_features.tsv: Top features with target variable<br>
{prefix}_train.distribution.txt: Class distribution<br>
{prefix}_train.scores.tsv: Prediction scores on training data<br>
<br>
Evaluation<br>
{prefix}_LearningCurves.pdf: Learning curves<br>
{prefix}_test.{target}.roc.cv.pdf: Cross-validation ROC curves<br>
{prefix}_test.AUC.cv.tsv: Cross-validation AUC scores<br>
<br>
Prediction<br>
{prefix}_predict.scores.tsv: Prediction results on new data<br>
{prefix}_validation.scores.tsv: Validation results (when using -m with -g)<br>
<br>
Correlation Analysis<br>
top_corr_data.txt: Top correlated features with target<br>
R_and_pValue.txt: Correlation coefficients and p-values<br>
corr.pdf: Correlation heatmap<br>
<br>
Causal Inference<br>
edges.txt: Causal relationships<br>
node_data.txt: Data for nodes related to target<br>
causal_graph.pdf: Visualized causal graph<br>
<br>
<h2><mark><strong>🤝 Contributing</strong></mark></h2>  
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.<br>
<br>
<h2><mark><strong>📄 License</strong></mark></h2>  
This project is licensed under the MIT License.<br>

<h2><mark><strong>👤 Author</strong></mark></h2>  
Chen Jing (cjwell@163.com/chenjing@matrix.com)<br>
<br>
<h2><mark><strong>中文</strong></mark></h2>  
<h2><mark><strong>📖 简介</strong></mark></h2>  
ModelFlow 是一个基于 scikit-learn 的综合机器学习工具，支持多种分类和回归算法。提供从数据预处理、模型训练、超参数优化到结果可视化和因果推断的端到端功能，让机器学习流程顺畅高效。<br>

<h2><mark><strong>✨ 功能特点</strong></mark></h2>  
多种算法：随机森林、决策树、逻辑回归、支持向量机、多层感知机、K近邻、朴素贝叶斯、梯度提升、AdaBoost、线性判别分析、Lasso<br>
数据预处理：自动独热编码、缺失值处理、特征相关性分析<br>
模型优化：网格搜索和随机搜索进行超参数优化<br>
特征重要性：内置重要性和排列重要性分析<br>
模型评估：ROC曲线、学习曲线、交叉验证、多种指标（AUC、准确率、F1、R2、MSE、MAE）<br>
可视化：相关性热图、ROC曲线、学习曲线、特征重要性图<br>
因果推断：基于PC算法的因果发现（需要causallearn包）<br>
模型持久化：使用joblib保存/加载模型<br>
预测：对新数据应用训练好的模型<br>
<br>
<h2><mark><strong>🛠️ 安装</strong></mark></h2>  
基础依赖<br>
pip install pandas matplotlib numpy scikit-learn scipy seaborn joblib lime shap<br>
<br>
可选依赖（用于因果推断）<br>
pip install causallearn pydot<br>
<br>
<h2><mark><strong>🚀 使用方法</strong></mark></h2>  
基本命令结构<br>
python ModelFlow.py -i 特征表.txt -g 表型文件.txt [选项]<br>
<br>
必需参数<br>
-i, --feature_table：输入特征表（行=样本，列=特征）<br>
-g, --phenotype：表型分组文件（第一列=样本ID，第二列=分组）<br>
<br>
可选参数<br>
参数	说明	默认值<br>
-it, --feature_transpose	如果样本在列则转置特征表	False<br>
-gc, --group_column	使用表型文件的第几列（从1开始）	1<br>
-gp, --group_pick	选择特定分组（格式：列号+关键词）	None<br>
-m, --model	加载已有模型进行预测	None<br>
-o, --output	输出目录	./<br>
-p, --prefix	输出文件前缀	方法名<br>
-mtd, --method	算法 (RF/DT/LGR/SVM/MLP/KNN/NBS/GBS/ABS/LDA/lasso)	RF<br>
-mn, --minnum	每类最小样本数	50<br>
-s, --scoring	评价指标	None<br>
-ta, --target	ROC曲线的目标类别	None<br>
-tb, --table	用于模型预测的特征重要性文件	None<br>
-opt, --optimize	超参数搜索 (grid/random/None)	random<br>
-cc, --cf_counts	学习曲线的最小样本数	100<br>
-mtdC, --method_Case	因果推断方法 (PC)	None<br>
-C1, --Case_pam1	因果推断的top特征数	30<br>
-mtdCor, --method_Cor	相关性分析 (SP)	None<br>
-Cor1, --Cor_pam1	相关性分析的top特征数	30<br>
<br>
使用示例<br>
训练随机森林模型：<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtd RF -o ./results -p my_project<br>
<br>
网格搜索超参数优化：<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtd SVM -opt grid -o ./results<br>
<br>
对新数据应用已有模型：<br>
python ModelFlow.py -i new_features.tsv -m trained_model.pkl -tb feature_importance.tsv -o ./predictions<br>
<br>
进行相关性分析：<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtdCor SP -Cor1 20 -o ./correlation<br>
<br>
因果推断（需要causallearn）：<br>
python ModelFlow.py -i features.tsv -g phenotype.tsv -mtdC PC -C1 30 -o ./causal<br>
<br>
<h2><mark><strong>📁 输出文件</strong></mark></h2>  
模型训练<br>
{前缀}_model.pkl：训练好的模型（包含特征和训练数据）<br>
{前缀}_best.params.txt：最佳超参数<br>
{前缀}_importances.tsv：特征重要性得分<br>
{前缀}_top{数量}_features.tsv：包含目标变量的top特征<br>
{前缀}_train.distribution.txt：类别分布<br>
{前缀}_train.scores.tsv：训练数据上的预测得分<br>
<br>
评估<br>
{前缀}_LearningCurves.pdf：学习曲线<br>
{前缀}_test.{目标}.roc.cv.pdf：交叉验证ROC曲线<br>
{前缀}_test.AUC.cv.tsv：交叉验证AUC得分<br>
预测<br>
{前缀}_predict.scores.tsv：新数据的预测结果<br>
{前缀}_validation.scores.tsv：验证结果（当同时使用-m和-g时）<br>
<br>
相关性分析<br>
top_corr_data.txt：与目标top相关的特征<br>
R_and_pValue.txt：相关系数和p值<br>
corr.pdf：相关性热图<br>
<br>
因果推断<br>
edges.txt：因果关系<br>
node_data.txt：与目标相关的节点数据<br>
causal_graph.pdf：可视化的因果图<br>
<br>
<h2><mark><strong>🤝 贡献</strong></mark></h2>  
欢迎贡献！请随时提交拉取请求或针对bug和功能需求开issue。<br>
<br>
<h2><mark><strong>📄 许可证</strong></mark></h2> 
本项目采用MIT许可证。<br>

<h2><mark><strong>👤 作者</strong></mark></h2>  
陈璟 (cjwell@163.com/chenjing@matridx.com)<br>
