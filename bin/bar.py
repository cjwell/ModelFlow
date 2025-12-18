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
import numpy as np
import matplotlib.pyplot as plt
def read_input_file(inp_file, CommonArea):

    if inp_file.endswith('.biom'):              #*  If the file format is biom:
        CommonArea = biom_processing(inp_file)  #*  Process in biom format
        return CommonArea                       #*  And return the CommonArea

    with open(inp_file) as inp:
        CommonArea['ReturnedData'] = [[v.strip() for v in line.strip().split("\t")] for line in inp.readlines()]
        return CommonArea

def transpose(data):
    return zip(*data)

def read_params(args):
    parser = argparse.ArgumentParser(description="bar图绘制，-i tab格式的文件 -g 1,2,3  -t all -o 输出文件路径  -p输出文件名.")
    parser.add_argument("-i", "--input_table", required=True, help="输入数据队列，使用列")
    parser.add_argument("-it", "--input_transpose", required=False, action='store_true', help="颠置输入队列")
    parser.add_argument("-g", "--group_column", required=False, type=lambda x: x.split(','),help="x,y,col的列名或者列号")
    parser.add_argument("-yn", "--y_noname", required=False, action='store_true',help="不显示y名字")
    parser.add_argument("-t", "--top_n", required=False, type=int,default=999,help="绘制y的数目，默认999")
    parser.add_argument("-bc", "--bar_color", required=False,  type=lambda x: x.split(','),default="red,blue" ,help="绘制颜色，默认red,blue")
    parser.add_argument("-o", "--output", required=False, default="./" , help="Output directory")
    parser.add_argument("-p", "--prefix", required=False,  help="File name prefix")
    parser.add_argument("-wh", "--width_height", required=False,  type=lambda x: x.split(','),default="10,10" , help="输出图片的宽高,默认10，10")
    args = parser.parse_args()

    return vars(args)



##bar图
def plot_bars(df,params,output_file):  ##数据框，数值x，类别y，类别col，y的数目，col的颜色盘，输出文件
    ##检查参数
    top_n=params['top_n']
    colors=params['bar_color']
    [w,h]=params['width_height']
    [x,y,col]=params['group_column']
    ###检查输入
    resolved_columns = []
    for cl in [x,y,col]:
        if cl.isdigit():  # 如果是数字，按列号转换为列名
            col_index = int(cl) - 1  # 这里假设列号从 1 开始
            if col_index >= 0 and col_index < df.shape[1]:  # 检查是否在范围内
                resolved_columns.append(df.columns[col_index])
            else:
                raise ValueError(f"列号 {cl} 超出范围")
        elif cl in df.columns:  # 如果是列名，直接添加
            resolved_columns.append(cl)
        else:
            raise ValueError(f"无效的列名或列号: {cl}")
    #列名
    [x,y,col]=resolved_columns
    print([x,y,col])
    if not pd.api.types.is_numeric_dtype(df[x]):
        print(f"{x}必须输入数值型")
        sys.exit()

    # 确保传入的列名存在于数据框中
    if x not in df.columns or y not in df.columns or col not in df.columns:
        raise ValueError("指定的列名在数据框中不存在")

    ##根据col过滤df
    df.dropna(subset=[col])
    grouped_df=df.copy()
    # 获取唯一分类变量和颜色映射
    unique_categories = grouped_df[col].unique()
    color_map = {cat: colors[i] for i, cat in enumerate(sorted(unique_categories))}
     #增加一列 如果分类变量数量超过颜色队列长度，使用白色
    default_color = 'white'
    grouped_df['color'] = grouped_df[col].map(color_map).fillna(default_color)
    
    ###多个取值时使用平均值，第一个颜色
    grouped_df = grouped_df.groupby(y).agg(
        {x: 'mean', 'color': 'first'}).reset_index()

    # 根据列2的值从高到低排序，并选择前top_n个
    sorted_df = grouped_df.nlargest(top_n, x)

    # 绘制条形图
    w=int(w)
    h=int(h)
    plt.figure(figsize=(w,h))

    plt.barh(sorted_df[y], sorted_df[x], color=sorted_df['color'],edgecolor='black')
    plt.xlabel('Values of ' + x)
    if params['y_noname'] is not True:
        plt.ylabel(y)
    plt.title(f'Top {top_n} {y} by {x}')
    plt.gca().invert_yaxis()  # 使得值高的条形在上方
     # 调整子图参数以保留图例空间
    plt.subplots_adjust(right=0.8)  # 增加右边的空白,值需要大于left
    plt.subplots_adjust(left=0.3)  # 增大左边空白，值可以根据需要调整

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[cat], edgecolor='black') for cat in unique_categories]
    plt.legend(handles, unique_categories, title='', bbox_to_anchor=(1.05, 1), loc='upper left',
            fontsize=12,           # Increase font size
            frameon=True,           # Enable frame (border)
            framealpha=1,           # Set border transparency (1 for opaque)
            edgecolor='black',      # Set border color
            borderpad=1,            # Space between legend content and border
            handlelength=1.5)       # Length of the legend handles

    plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()
	   


def plot_stacked_bar(df, output_file):
    """
    绘制堆叠柱状图，展示不同组的微生物丰度，并确保组与组之间有明显分割
    """

    # 获取微生物种类列表
    microbes = df['microbe'].unique()
    
    # 获取样本分组
    groups = df['group'].unique()

    # 获取样本列表（所有样本，假设每个分组的样本数相同）
    samples = df['sample'].unique()

    # 为每个微生物种类指定颜色
    colors = plt.cm.get_cmap("tab10", len(microbes))

    # 创建图形和子图
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.8  # 设置柱子的宽度
    x_gd=0
    # 遍历每个微生物种类，绘制堆叠柱状图
    for i, group in enumerate(groups):
        ##提取子集
        group_data = df[(df['group'] == group)]
        sample = group_data['sample'].unique()
        # 调整样本位置使得每个分组有不同的x坐标（并保持柱状图对齐）
        x_offset = np.arange(len(sample)) +  (bar_width + 0.2) +x_gd # 每个组的x坐标
        x_gd+=len(sample)
        bottom_values = np.zeros(len(sample))  # 堆叠柱的底部初始为零
        print(sample)
        print(bottom_values)
        print(x_offset)
        for j, microbe in enumerate(microbes):
            gd = group_data[(group_data['microbe'] == microbe)]
            abundance = gd['abundance'].values        
            ax.bar(x_offset, abundance, width=bar_width, bottom=bottom_values, color=colors(j), label=f'{microbe} - {group}')
            bottom_values += abundance  # 更新底部值


    # 设置x轴标签、y轴标签、图例等
    ax.set_xticks(np.arange(len(samples)) + (len(groups)-1) * (bar_width + 0.2) / 2)
    ax.set_xticklabels(samples)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Abundance')
    ax.set_title('Microbial Abundance across Groups')
    
    # 设置图例
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # 调整布局
    plt.tight_layout()

    # 保存图形为PDF文件
    plt.savefig(output_file, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()





def main():
    params = read_params(sys.argv)

    # 读取input
    try:
        df = pd.read_csv(params['input_table'],sep='\t', dtype={'column_name': str},low_memory=False)
        print("DataFrame loaded successfully:")
        print(df.head())
    except Exception as e:
        print(f"Error reading file: {e}")
    if params['input_transpose'] is not False:
        df = df.T

    ##判定输出
    if params['prefix'] is None:
        params['prefix']  = os.path.splitext(os.path.basename(params['input_table']))[0]
    output_file = os.path.join(params['output'], f"{params['prefix']}.bar.pdf")

    # 创建输出目录（如果不存在）
    os.makedirs(params['output'], exist_ok=True)

    #plot_bars(df,params,output_file)

    # 创建示例数据框
    data = {
        'sample': [f'Sample {i+1}' for i in range(20)] * 5,
        'microbe': ['Microbe 1']*20+ ['Microbe 2']*20+ ['Microbe 3']*20+['Microbe 4']*20+['Microbe 5'] * 20,
        'abundance': np.random.rand(100),
        'group': ['Group 1'] * 10 + ['Group 2'] * 10+['Group 1'] * 10 + ['Group 2'] * 10+['Group 1'] * 10 + ['Group 2'] * 10+['Group 1'] * 10 + ['Group 2'] * 10+['Group 1'] * 10 + ['Group 2'] * 10
    }
    df = pd.DataFrame(data)
    print(df)
    plot_stacked_bar(df,output_file)

if __name__ == "__main__":
    main()