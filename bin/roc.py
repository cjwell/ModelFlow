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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

def plot_roc(df,score,target,colors,output_pdf,w=8,h=8):
    """
    生成并保存指定类别的ROC曲线。
    参数:
    - tg: 指定的目标类别（df中的某一类别）
    - output_file: 输出文件全路径

    """
    ###检查输入
    resolved_columns = []
    for col in [score,target]:
        if col.isdigit():  # 如果是数字，按列号转换为列名
            col_index = int(col) - 1  # 这里假设列号从 1 开始
            if col_index >= 0 and col_index < df.shape[1]:  # 检查是否在范围内
                resolved_columns.append(df.columns[col_index])
            else:
                raise ValueError(f"列号 {col} 超出范围")
        elif col in df.columns:  # 如果是列名，直接添加
            resolved_columns.append(col)
        else:
            raise ValueError(f"无效的列名或列号: {col}")
    #列名
    [score,target]=resolved_columns
    print([score,target])
    if not pd.api.types.is_numeric_dtype(df[score]):
        print(f"{score}必须输入数值型")
        sys.exit()

    # 确保传入的列名存在于数据框中
    if score not in df.columns or target not in df.columns:
        raise ValueError("指定的列名在数据框中不存在")

    ##根据target过滤df
    df.dropna(subset=[target])

    # 根据列2的值从高到低排序，并选择前top_n个
    #sorted_df = df.nlargest(top_n, x)

    # 获取唯一分类变量和颜色映射
    unique_categories = sorted(df[target].unique())
    color_map = {cat: colors[i] for i, cat in enumerate(unique_categories)}
    print(color_map)
    
    # 如果分类变量数量超过颜色队列长度，使用白色
    #default_color = 'white'
    #df['color'] = df[col].map(color_map).fillna(default_color)

    # 绘制ROC曲线
    w=int(w)
    h=int(h)
    plt.figure(figsize=(w, h))
    auc_df=[]
    for idx,tg in enumerate(unique_categories):
        col=colors[idx]

        if tg is None:
            continue
        #子数据框
        sub_df = pd.DataFrame()    
        sub_df=df.copy()
            
        #新增一列
        target_class=target
        # 替换不是target的值为'other'
        sub_df.loc[sub_df[target] != tg, target] = 'other'


        ##子数据框的actual
        y=sub_df[target]
        y_scores=sub_df[score]

        fpr, tpr, _ = roc_curve(y, y_scores, pos_label=tg)
        roc_auc=auc(fpr,tpr)
        if roc_auc <0.5:
            tpr=1-tpr
            fpr=1-fpr
            roc_auc=1-roc_auc
        auc_df.append([target,tg,roc_auc])
        plt.plot(fpr, tpr, lw=3,color=col,label=f'{tg} (AUC = {roc_auc:.2f})')
        plt.legend(loc='lower right',
            fontsize=20,           # Increase font size
            frameon=True,           # Enable frame (border)
            framealpha=1,           # Set border transparency (1 for opaque)
            edgecolor='black',      # Set border color
            borderpad=1,            # Space between legend content and border
            handlelength=1.5)       # Length of the legend handles
    # 图表设置
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title(f"ROC curve of {score}",fontsize=30)
    plt.grid()

    # 保存图像
    plt.savefig(output_pdf, format='pdf')
    plt.close()
    print(f'ROC 曲线保存在 {output_pdf}')
    ##auc输出
    auc_df = pd.DataFrame(auc_df, columns=['run', 'target', 'AUC'])
    return auc_df
	   

def main():
    parser = argparse.ArgumentParser(description="bar图绘制，-i tab格式的文件 -g 1,2,3  -t all -o 输出文件路径  -p输出文件名.")
    parser.add_argument("-i", "--input_table", required=True, help="输入数据队列，使用列")
    parser.add_argument("-it", "--input_transpose", required=False, action='store_true', help="颠置输入队列")
    parser.add_argument("-g", "--group_column", required=True, type=lambda x: x.split(','),help="score,target的列名或者列号,score为数值，target为分类")
    parser.add_argument("-t", "--top_n", required=False, type=int,default=999,help="绘制y的数目，默认999")
    parser.add_argument("-bc", "--bar_color", required=False,  type=lambda x: x.split(','),default="red,blue" ,help="绘制颜色，默认 red，blue")
    parser.add_argument("-o", "--output", required=False, default="./" , help="Output directory")
    parser.add_argument("-p", "--prefix", required=False,  help="File name prefix")
    parser.add_argument("-wh", "--width_height", required=False,  type=lambda x: x.split(','),default="10,10" , help="输出图片的宽高，默认10，10")
    args = parser.parse_args()

    # 读取input
    try:
        df = pd.read_csv(args.input_table,sep='\t', dtype={'column_name': str},low_memory=False)
        print("DataFrame loaded successfully:")
        print(df.head())
    except Exception as e:
        print(f"Error reading file: {e}")
    if args.input_transpose is not False:
        df = df.T

    ##判定输出
    if args.prefix is None:
        args.prefix = os.path.splitext(os.path.basename(args.input_table))[0]
    output_file = os.path.join(args.output, f"{args.prefix}_roc.pdf")
    ##检查参数
    colors=args.bar_color
    [w,h]=args.width_height
    [score,target]=args.group_column
    # 创建输出目录（如果不存在）
    os.makedirs(args.output, exist_ok=True)

    plot_roc(df,score,target,colors,output_file,w,h)

if __name__ == "__main__":
    main()