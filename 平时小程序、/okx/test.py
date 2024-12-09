
# 库文件
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from scipy.optimize import NonlinearConstraint
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 设置中文字体
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 替换为你的中文字体文件路径
import sys
sys.path.append(r"C:\Users\Admin\Documents\GitHub\gaolu\MPC\高炉")
import optuna
import numpy as np
import optuna
import numpy as np
import base 
# 库文件
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import tensorflow as tf
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 设置中文字体
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 替换为你的中文字体文件路径
import sys
sys.path.append(r"C:\Users\Admin\Documents\GitHub\gaolu\MPC\高炉")
from collections import deque

import base 
# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import os
import datetime
import pickle
# 机器学习库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
# 数据归一化、逆归一化
from sklearn.preprocessing import MinMaxScaler
# 优化相关库
from skopt import gp_minimize
from scipy.optimize import minimize

# 深度学习库
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 中文字体设置
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 替换为你的中文字体文件路径

# 其他路径设置
sys.path.append(r"C:\Users\Admin\Documents\GitHub\gaolu\MPC\高炉")

import torch
 
# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 读取Excel文件
# C:\Users\Admin\Documents\GitHub\gaolu\MPC\高炉\0数据处理\新输入输出模式\1h_mean.xlsx
excel_path = f'C:\\Users\\Admin\\Documents\\GitHub\\gaolu\\MPC\\高炉\\0数据处理\\新输入输出模式\\1h_mean.xlsx'
df_sheet_yuansu = pd.read_excel(excel_path, sheet_name='原始输出') 
# df_sheet_yuansu = pd.read_excel(excel_path, sheet_name='剔除直线输出') 
# df_sheet_yuansu = pd.read_excel(excel_path, sheet_name='单SI_0.2_0.8') 
# print(df_sheet_yuansu.info())
# print(df_sheet_yuansu.columns)

excel_path = f'C:\\Users\\Admin\\Documents\\GitHub\\gaolu\\MPC\\高炉\\0数据处理\\新输入输出模式\\1h_mean.xlsx'
df_sheet_params = pd.read_excel(excel_path, sheet_name='1h_mean_all') 

# print(df_sheet_params.info())
# print(df_sheet_params.columns)




# 检查 DataFrame 中是否包含 NaN 值
def check_if_NaN(data):
    print(data.shape)
    contains_nan = data.isna().any().any()
    if contains_nan:
        print("数据包含 NaN 值")
    else:
        print("数据不包含 NaN 值")

        
check_if_NaN(df_sheet_yuansu)
check_if_NaN(df_sheet_params)



df_sheet_params.columns



input_term =        ['富氧流量', '冷风流量', '热风压力', '热风温度']
last_input_term =        ['富氧流量2', '设定喷煤量2', '热风压力2', '热风温度2']
output_term = ['铁水温度[MIT]', '铁水硅含量[SI]']
last_output_term = ['铁水温度[MIT]2', '铁水硅含量[SI]2']
time_term= '时间戳h'




















# 处理异常值

# 创建数据框副本以避免修改原始数据

df_sheet_X = df_sheet_params.copy()
df_sheet_X_process = df_sheet_X.copy()
df_sheet_Y = df_sheet_yuansu.copy()
df_sheet_Y_process = df_sheet_Y.copy()


def IQR_process(df_IQR, columns):
    df_IQR = df_IQR
    columns = columns

    print(columns)      # 获取数据框的所有列名
    outlier_indices = set()  # 用于存储异常值的行索引

    # 1. 分别处理每个变量
    for column in columns:
        # 计算描述性统计
        stats = df_IQR[column].describe()

        # 计算IQR（四分位距）以及上下须的范围
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        lower_whisker = Q1 - 8 * IQR
        upper_whisker = Q3 + 8 * IQR
        # if column == '热风压力':
        #     lower_whisker = Q1 - 2 * IQR
        #     upper_whisker = Q3 + 1.5 * IQR

        # # 绘制箱线图
        # plt.figure(figsize=(8, 6))
        # sns.boxplot(data=df_IQR[column])
        # plt.title(f'Boxplot of {column}', fontproperties=font)
        # plt.xlabel('Feature', fontproperties=font)
        # plt.ylabel('Value', fontproperties=font)
        # plt.show()

        # 查找异常值的索引
        outliers = df_IQR[(df_IQR[column] < lower_whisker) | 
                            (df_IQR[column] > upper_whisker)].index
        outlier_indices.update(outliers)

        # # 打印统计信息和异常值范围
        # print(f"列: {column}")
        # print(f"第一四分位数 (Q1): {Q1}")
        # print(f"第三四分位数 (Q3): {Q3}")
        # print(f"下须 (lower whisker): {lower_whisker}")
        # print(f"上须 (upper whisker): {upper_whisker}")
        # print(f"找到的异常值索引: {list(outliers)}")

        
        # print(f"异常值数量: {len(outliers)}")
        # print(f"总数: {len(df_IQR[column])}")

        # print(f"异常值比例: {len(outliers)/len(df_IQR[column])}\n")

    # 2. 删除所有异常值
    df_cleaned = df_IQR.drop(index=outlier_indices)
    # 重新设置索引，使索引从 0 开始，并丢弃旧索引
    df_cleaned.reset_index(drop=True, inplace=True)
    # 输出处理后的数据框信息
    print(f"原始数据行数: {df_IQR.shape[0]}")
    print(f"删除异常值后的数据行数: {df_cleaned.shape[0]}")

    # 你可以继续对 df_cleaned 进行后续处理



    return df_cleaned


df_cleaned_X = IQR_process(df_sheet_X_process, input_term)
df_cleaned_Y = IQR_process(df_sheet_Y_process, output_term)

# print(np.max(df_cleaned_Y['铁水温度[MIT]']))
# print(np.min(df_cleaned_Y['铁水温度[MIT]']))
# print(np.max(df_cleaned_Y['铁水硅含量[SI]']))
# print(np.min(df_cleaned_Y['铁水硅含量[SI]']))







# 异常数据处理-处理前后对比
# 创建数据框副本以避免修改原始数据
df_sheet_yuansu_process = df_cleaned_Y.copy()
df_sheet_params_process = df_cleaned_X.copy()
# 定义一个函数，用前后两个值的差值按照距离进行加权替换异常值
def replace_outliers_with_weighted_diff(x, y):
    # 计算列的中位数
    median_value = y.median()
    # 检测异常值的索引
    outliers_index = (y - median_value).abs() > 2.5 * y.std()
    
    # 遍历异常值的索引
    for idx in outliers_index[outliers_index].index:
        # 获取异常值前一个和后一个值的索引
        prev_idx = idx - 1 if idx - 1 >= 0 else idx
        next_idx = idx + 1 if idx + 1 < len(y) else idx
        # 计算当前 x 与前后两个 x 的距离
        dist_prev = abs(x[idx] - x[prev_idx])
        dist_next = abs(x[next_idx] - x[idx])
        total_dist = dist_prev + dist_next
        # 计算权重
        weight_prev = dist_next / total_dist
        weight_next = dist_prev / total_dist
        # 计算前后两个值的差值
        diff = y[next_idx] - y[prev_idx]
        # 根据权重进行插值
        interpolated_value = y[prev_idx] + weight_prev * diff
        # 用插值结果替代异常值
        y[idx] = interpolated_value



# 对指定列应用替代异常值的函数
# 对指定列应用替代异常值的函数
replace_outliers_with_weighted_diff(df_sheet_params_process[time_term], df_sheet_params_process[input_term[0]])
replace_outliers_with_weighted_diff(df_sheet_params_process[time_term], df_sheet_params_process[input_term[1]])
replace_outliers_with_weighted_diff(df_sheet_params_process[time_term], df_sheet_params_process[input_term[2]])
replace_outliers_with_weighted_diff(df_sheet_params_process[time_term], df_sheet_params_process[input_term[3]])
# replace_outliers_with_weighted_diff(df_sheet_params_process[time_term], df_sheet_params_process[input_term[4]])
# replace_outliers_with_weighted_diff(df_sheet_params_process[time_term], df_sheet_params_process[input_term[5]])
# replace_outliers_with_weighted_diff(df_sheet_params_process[time_term], df_sheet_params_process[input_term[6]])

# replace_outliers_with_weighted_diff(df_sheet_yuansu_process[time_term], df_sheet_yuansu_process[output_term[0]])
# replace_outliers_with_weighted_diff(df_sheet_yuansu_process[time_term], df_sheet_yuansu_process[output_term[1]])




length1 = 400
start1 = 0
length2 = 400
start2 = 400


index_gaolu   = range(start1, start1+length1+1, 1)
index_predict     = range(start2, start2+length2+1, 1)
# index = range(1, 7572, 1)


df_sheet_yuansu_process.describe(percentiles=[.10, .90])



# 数据归一化、逆归一化
from sklearn.preprocessing import MinMaxScaler

# 将数据存储为字典，每个键对应一列数据
original_data_dict = {
    input_term[0]:   df_sheet_params_process[input_term[0]].values,
    input_term[1]:   df_sheet_params_process[input_term[1]].values,
    input_term[2]:   df_sheet_params_process[input_term[2]].values,
    input_term[3]:   df_sheet_params_process[input_term[3]].values,
    # input_term[4]:   df_sheet_params_process[input_term[4]].values,
    # input_term[5]:   df_sheet_params_process[input_term[5]].values,
    # input_term[6]:   df_sheet_params_process[input_term[6]].values,
    output_term[0]:  df_sheet_yuansu_process[output_term[0]].values,
    output_term[1]:  df_sheet_yuansu_process[output_term[1]].values
}

# 初始化缩放器
scalers = {}

# 进行拟合
for column, data in original_data_dict.items():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data.reshape(-1, 1))  # 保证数据是列向量
    scalers[column] = scaler

# 进行归一化
normalized_data_dict = {}
for column, scaler in scalers.items():
    normalized_data_dict[column] = scaler.transform(original_data_dict[column].reshape(-1, 1)).flatten()

# 进行反归一化
original_data_dict = {}
for column, scaler in scalers.items():
    original_data_dict[column] = scaler.inverse_transform(normalized_data_dict[column].reshape(-1, 1)).flatten()



# 标定归一化前后数据
data_point = np.array([1500]).reshape(-1, 1)
data1 = scalers[output_term[0]].transform(data_point).flatten()

data_point = np.array(data1).reshape(-1, 1)
data2 = scalers[output_term[0]].inverse_transform(data_point).flatten()

data_point = np.array([1510]).reshape(-1, 1)
data3 = scalers[output_term[0]].transform(data_point).flatten()

data_point = np.array(data3).reshape(-1, 1)
data4 = scalers[output_term[0]].inverse_transform(data_point).flatten()

print(data1)
print(data2)
print(data3)
print(data4)
d_temp = (data3-data1)/(data4-data2)
print('每摄氏度的输出差：',d_temp)



data_point = np.array([0.50]).reshape(-1, 1)
data1 = scalers[output_term[1]].transform(data_point).flatten()

data_point = np.array(data1).reshape(-1, 1)
data2 = scalers[output_term[1]].inverse_transform(data_point).flatten()

data_point = np.array([0.60]).reshape(-1, 1)
data3 = scalers[output_term[1]].transform(data_point).flatten()

data_point = np.array(data3).reshape(-1, 1)
data4 = scalers[output_term[1]].inverse_transform(data_point).flatten()

print(data1)
print(data2)
print(data3)
print(data4)
d_yuansu = (data3-data1)/(data4-data2)
print('每浓度的输出差：',(data3-data1))



isShuffle = True
isShuffle = False
time_steps = 2
test_size = 0.15
val_size = 0.15
train_size = 1-val_size-test_size



# 组合训练数据--拆分训练、测试集

# 定义时间步数和特征数

# 构成    
# X = [X(t),X(t-1),Y(t-1)]
# Y = [Y(t)]
def make_data(u1_data,u2_data,u3_data,u4_data,y1_data,y2_data,index_fanwei):
    X = np.column_stack((u1_data,u2_data,u3_data,u4_data))
    y = np.column_stack((y1_data, y2_data))

    X_modified = []
    y_modified = []
    
    for i in range(3,len(y1_data)):
        if i in index_fanwei:
            # print(i)
            # print(df_sheet_yuansu[time_term][i])
            yuansu_time = df_sheet_yuansu[time_term][i]
            closest_10 = df_sheet_params[df_sheet_params[time_term] <= yuansu_time].nlargest(time_steps, time_term)
            # print(closest_10)
            
            index = closest_10.index
            # print(index)
            # print(closest_10.iloc[-1][time_term])
            if closest_10.iloc[-1][time_term] < yuansu_time - time_steps + 1:
                print(i,yuansu_time,'errloss')
            else:

                # print(X[index, :])
                new_x_sample = np.concatenate([X[i, :] for i in index],axis=0)
                # print(new_x_sample)
                y_last = y[i-1, :]
                # print(y_last, 'y_last time : ',df_sheet_yuansu[time_term][i-1])
                new_x_sample = np.concatenate([new_x_sample,y_last],axis=0)
                # print(new_x_sample)
                y_sample = y[i, :]  
                X_modified.append(new_x_sample)
                y_modified.append(y_sample)
                print(i,yuansu_time,index[0],index[-1], end='\r')
                # break

    # 将列表转换为 NumPy 数组
    X_modified = np.array(X_modified)
    y_modified = np.array(y_modified)
    X_reshaped = X_modified.reshape((X_modified.shape[0], X_modified.shape[1]))

    # 打印新数据的形状
    print("Modified Input Shape:", X_reshaped.shape)
    print("Modified Output Shape:", y_modified.shape)


    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_modified, 
                                                        test_size=test_size, 
                                                        random_state=42, 
                                                        shuffle=isShuffle)

    # 将剩余的70%训练数据再次拆分成训练数据和验证数据（20%验证数据，50%训练数据）
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                        test_size=val_size/(train_size+val_size), 
                                                        random_state=42, 
                                                        shuffle=isShuffle)

    print('训练数量：',X_train.shape,y_train.shape)
    print('验证数量：',X_val.shape,y_val.shape)
    print('测试数量：',X_test.shape,y_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test



def symmetrical_moving_average(data, N):
    """
    使用对称的移动平均滤波，当前值由其自身及其前后的值决定。
    
    :param data: 输入的数据序列，一般为列表或者NumPy数组。
    :return: 经过滤波处理的数据序列。
    """
    filtered_data = []
    N = 9
    percent = 0.8
    # 遍历数据，从索引1开始到倒数第二个元素结束
    for i in range(1, len(data) - 1):
        # 计算当前值及其前后值的平均
        average = (data[i - 1]*(1-percent)/2 + data[i]*percent + data[i + 1]*(1-percent)/2)
        filtered_data.append(average)
    
    # 对于序列的第一个和最后一个元素，直接使用原始值
    # 或者可以使用其他边界处理策略
    filtered_data.insert(0, data[0])
    filtered_data.append(data[-1])
    
    return np.array(filtered_data)

# 示例数据
data = [2, 4, 6, 8, 10, 12, 14]
filtered_data = symmetrical_moving_average(data,9)
print(filtered_data)





# 高炉模型列数据
u1_data = normalized_data_dict[input_term[0]]
u2_data = normalized_data_dict[input_term[1]]
u3_data = normalized_data_dict[input_term[2]]
u4_data = normalized_data_dict[input_term[3]]
y1_data = normalized_data_dict[output_term[0]]
y2_data = normalized_data_dict[output_term[1]]
num_samples = y2_data.shape[0]

# filter_windows = 2
# u1_data = symmetrical_moving_average(u1_data, filter_windows)
# u2_data = symmetrical_moving_average(u2_data, filter_windows)
# u3_data = symmetrical_moving_average(u3_data, filter_windows)
# u4_data = symmetrical_moving_average(u4_data, filter_windows)
# u5_data = symmetrical_moving_average(u5_data, filter_windows)
# u6_data = symmetrical_moving_average(u6_data, filter_windows)
# u7_data = symmetrical_moving_average(u7_data, filter_windows)
# y1_data = symmetrical_moving_average(y1_data, filter_windows)
# y2_data = symmetrical_moving_average(y2_data, filter_windows)

print('高炉模型数据')
X_gaolu_train, X_gaolu_val, X_gaolu_test,\
y_gaolu_train, y_gaolu_val, y_gaolu_test = make_data(u1_data,u2_data,u3_data,u4_data,
                                                            y1_data,y2_data,
                                                            index_fanwei=index_gaolu)



# 预测模型列数据
u1_data = normalized_data_dict[input_term[0]]
u2_data = normalized_data_dict[input_term[1]]
u3_data = normalized_data_dict[input_term[2]]
u4_data = normalized_data_dict[input_term[3]]
y1_data = normalized_data_dict[output_term[0]]
y2_data = normalized_data_dict[output_term[1]]
num_samples = y2_data.shape[0]

# filter_windows = 2
# u1_data = symmetrical_moving_average(u1_data, filter_windows)
# u2_data = symmetrical_moving_average(u2_data, filter_windows)
# u3_data = symmetrical_moving_average(u3_data, filter_windows)
# u4_data = symmetrical_moving_average(u4_data, filter_windows)
# u5_data = symmetrical_moving_average(u5_data, filter_windows)
# u6_data = symmetrical_moving_average(u6_data, filter_windows)
# u7_data = symmetrical_moving_average(u7_data, filter_windows)
# y1_data = symmetrical_moving_average(y1_data, filter_windows)
# y2_data = symmetrical_moving_average(y2_data, filter_windows)
print('预测模型数据')
X_predict_train, X_predict_val, X_predict_test,\
y_predict_train, y_predict_val, y_predict_test = make_data(u1_data,u2_data,u3_data,u4_data,
                                                            y1_data,y2_data,
                                                            index_fanwei=index_predict)


# 假设make_data返回的是NumPy数组，我们需要将它们转换为PyTorch张量
# 注意：如果make_data已经返回了张量，则这一步可以省略
def to_tensor(data, device):
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32).to(device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError("Unsupported data type")
 
# 对高炉模型数据进行设备转换
X_gaolu_train = to_tensor(X_gaolu_train, device)
X_gaolu_val = to_tensor(X_gaolu_val, device)
X_gaolu_test = to_tensor(X_gaolu_test, device)
y_gaolu_train = to_tensor(y_gaolu_train, device)
y_gaolu_val = to_tensor(y_gaolu_val, device)
y_gaolu_test = to_tensor(y_gaolu_test, device)
 
# 对预测模型数据进行设备转换（如果它们的结构和处理与高炉模型数据相同）
X_predict_train = to_tensor(X_predict_train, device)
X_predict_val = to_tensor(X_predict_val, device)
X_predict_test = to_tensor(X_predict_test, device)
y_predict_train = to_tensor(y_predict_train, device)
y_predict_val = to_tensor(y_predict_val, device)
y_predict_test = to_tensor(y_predict_test, device)
 
# 现在，你的数据已经在正确的设备上了，可以传递给模型进行训练或预测



print('高炉模型数据')
X_1, X_2, X_3,\
y_1, y_2, y_3 = make_data(u1_data,u2_data,u3_data,u4_data,
                                                            y1_data,y2_data,
                                                            index_fanwei=range(0, 7000, 1))






X_concat = np.concatenate((X_1, X_2, X_3), axis=0)
print(X_concat.shape)
y_concat = np.concatenate((y_1, y_2, y_3), axis=0)
print(y_concat.shape)
new_concat = np.concatenate((X_concat[1:,0:4], y_concat[1:,0:2], y_concat[:-1,0:2]), axis=1)

print(new_concat.shape)
print(X_concat[0:2,:])
print(y_concat[0:2,:])
print(new_concat[0:2,:])



data2_all = {
    input_term[0]: new_concat[1:,0],
    input_term[1]: new_concat[1:,1],
    input_term[2]: new_concat[1:,2],
    input_term[3]: new_concat[1:,3],
    last_input_term[0]: new_concat[:-1,0],
    last_input_term[1]: new_concat[:-1:,1],
    last_input_term[2]: new_concat[:-1:,2],
    last_input_term[3]: new_concat[:-1:,3],
    output_term[0]: new_concat[1:,4],
    output_term[1]: new_concat[1:,5],
    last_output_term[0]: new_concat[1:,6],
    last_output_term[1]: new_concat[1:,7]
}

# 将字典转换为 DataFrame
data2_all = pd.DataFrame(data2_all)

# 查看生成的 DataFrame
print(data2_all)



ghgjh = 0
all_rand_num = []
all_pred_y1_mse = []
all_pred_y2_mse = []


for ghgjh in range(1949,20000,1):


    epoch_once_time = 50
    ischuangxin = True
    # ischuangxin = False
    cengshu = 3



    # 定义模型
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class MyNeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, if_chuangxin = False,gamma = 0.1):
            self.if_chuangxin = if_chuangxin
            super(MyNeuralNetwork, self).__init__()
            if cengshu == 3:    
                if self.if_chuangxin:            
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_size, hidden_size)
                    self.fc3 = nn.Linear(hidden_size, hidden_size)
                    self.fc4 = nn.Linear(hidden_size, output_size)
                else:
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, hidden_size)
                    self.fc3 = nn.Linear(hidden_size, hidden_size)
                    self.fc4 = nn.Linear(hidden_size, output_size)
                    self.relu = nn.ReLU()
            elif cengshu == 2:  
                if self.if_chuangxin:            
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_size, hidden_size)
                    self.fc3 = nn.Linear(hidden_size, output_size)
                else:
                    self.fc1 = nn.Linear(input_size, hidden_size)
                    self.fc2 = nn.Linear(hidden_size, hidden_size)
                    self.fc3 = nn.Linear(hidden_size, output_size)
                    self.relu = nn.ReLU()


        def forward(self, x0):
            if cengshu == 3:    
                if self.if_chuangxin:

                    x = self.fc1(x0)
                    x = self.relu(x)

                    x2 = self.fc2(x)
                    x2 = self.relu(x2)

                    x3 = self.fc3(x2)
                    x3 = self.relu(x3)

                    x4 = x + x2 + x3
                    output = self.fc4(x4)
                else:
                    x = self.fc1(x0)
                    x = self.relu(x)

                    x2 = self.fc2(x)
                    x2 = self.relu(x2)

                    x3 = self.fc3(x2)
                    x3 = self.relu(x3)

                    output = self.fc4(x3)
            elif cengshu == 2:  
                if self.if_chuangxin:

                    x = self.fc1(x0)
                    x = self.relu(x)

                    x2 = self.fc2(x)
                    x2 = self.relu(x2)

                    x3 = x + x2
                    output = self.fc3(x3)
                else:
                    x = self.fc1(x0)
                    x = self.relu(x)

                    x2 = self.fc2(x)
                    x2 = self.relu(x2)

                    output = self.fc3(x2)
            return output
            
            return output
        
        

        
        def custom_loss(self, y_true, y_pred):

            squared_diff = torch.pow(y_true - y_pred, 2)
            sum_squared_diff = torch.sum(squared_diff)
            mse = sum_squared_diff / len(y_true)
            return mse
        

        def my_fit(self, 
                    X_train, y_train, 
                    X_val, y_val, 
                    train_loss_list,val_loss_list,
                    epochs=1, batch_size=32, lr=0.001):
            optimizer = optim.Adam(self.parameters(), lr=lr)


            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, len(X_train), batch_size):
                    x_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
                    y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)

                    optimizer.zero_grad()
                    y_pred = self(x_batch)
                    loss = self.custom_loss(y_batch, y_pred)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                average_epoch_train_loss = epoch_loss / (len(X_train) / batch_size)
                # 验证集评估
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    for i in range(0, len(X_val), batch_size):
                        x_batch_val = torch.tensor(X_val[i:i+batch_size], dtype=torch.float32)
                        y_batch_val = torch.tensor(y_val[i:i+batch_size], dtype=torch.float32)

                        y_pred_val = self(x_batch_val)
                        val_loss += self.custom_loss(y_batch_val, y_pred_val).item()

                    average_epoch_val_loss = val_loss / (len(X_val) / batch_size)

                print(f'第 {epoch + 1}/{epochs} 轮, 训练误差: {average_epoch_train_loss:.4f}, 验证误差: {average_epoch_val_loss:.4f}', end='\r')
                train_loss_list.append(average_epoch_train_loss)
                val_loss_list.append(average_epoch_val_loss)

            return train_loss_list,val_loss_list
        
        def model_update(self, 
                    X_train, y_train, 
                    epochs=1, batch_size=32, lr=0.001,ifprint = False):
            optimizer = optim.Adam(self.parameters(), lr=lr)
            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, len(X_train), batch_size):
                    x_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
                    y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)

                    optimizer.zero_grad()
                    y_pred = self(x_batch)
                    loss = self.custom_loss(y_batch, y_pred)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                average_epoch_train_loss = epoch_loss / (len(X_train) / batch_size)
                if(ifprint):print(f'第 {epoch + 1}/{epochs} 轮, 训练误差: {average_epoch_train_loss:.4f}')
                
                
            return 0
        
        

        def my_predict(self, X_test):
            # 设置模型为评估模式，这会关闭 dropout 等层
            self.eval()
            # 将输入数据转换为张量，并设置 requires_grad=True
            x_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
            
            # 获取模型的预测输出
            y_pred = self(x_tensor)
            # 保留预测值的梯度信息
            y_pred.retain_grad()
            # 返回预测结果和包含梯度信息的张量

            # return y_pred[:,0].detach().numpy(),y_pred[:,1].detach().numpy()
        
            # 将张量从CUDA移动到CPU，然后再转换为NumPy数组
            y_pred_cpu = y_pred.detach().cpu()  # 这一步是新增的，用于将张量移动到CPU
            return y_pred_cpu[:, 0].numpy(), y_pred_cpu[:, 1].numpy()  # 现在可以安全地转换为NumPy数组了




    # 建立高炉模型实例
    input_size = 10  # 输入特征大小
    hidden_size = 16  # 32
    output_size = 2  # 输出大小
    # 设置随机种子
    torch.manual_seed(0)
    model_gaolu = MyNeuralNetwork(input_size, 
                                hidden_size,
                                output_size,
                                ischuangxin,
                                gamma = 0.1).to(device) 
    epoch_sum_gaolu = 0
    gaolu_train_loss_list = []
    gaolu_val_loss_list = []



    # 高炉模型训练
    epoch_once = epoch_once_time
    epoch_sum_gaolu = epoch_sum_gaolu + epoch_once
    gaolu_train_loss_list,gaolu_val_loss_list = model_gaolu.my_fit(X_gaolu_train, y_gaolu_train,
                                        X_gaolu_val, y_gaolu_val, 
                                        gaolu_train_loss_list, gaolu_val_loss_list,
                                        epochs=epoch_once, 
                                        batch_size=32,
                                        lr = 0.002)

    print('\nepoch_sum:',epoch_sum_gaolu)






    input_term



    # 用于子图编号的字母序列
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    input_term333 =       input_term
    output_term333 = ['铁水温度MIT', '铁水硅含量[Si]']
    time_term= '时间戳h'
    print(input_term333)
    print(output_term333)
    input_term222 =        ['富氧流量/(m\u00b3/h)', '冷风流量/(m\u00b3/h)', '热风压力/kPa', '热风温度/℃']
    output_term222 = ['铁水温度MIT/℃', '铁水硅含量[Si]/%']
    time_term= '时间戳h'
    print(input_term222)
    print(output_term222)



    ditem = -0.65



    # 高炉模型建模效果
    y_train_pred_0,y_train_pred_1 = model_gaolu.my_predict(X_gaolu_train)
    y_test_pred_0,y_test_pred_1 = model_gaolu.my_predict(X_gaolu_test)

 


    # 创建预测模型实例
    # 设置随机种子
    torch.manual_seed(0)
    model_predict = MyNeuralNetwork(input_size, hidden_size, output_size,ischuangxin).to(device) 
    epoch_sum_predict = 0
    predict_train_loss_list = []
    predict_val_loss_list = []



    # 预测模型训练
    epoch_once = epoch_once_time
    epoch_sum = epoch_sum_predict + epoch_once
    predict_train_loss_list, predict_val_loss_list = model_predict.my_fit(X_predict_train, y_predict_train,
                                        X_predict_val, y_predict_val, 
                                        predict_train_loss_list, predict_val_loss_list,
                                        epochs=epoch_once, 
                                        batch_size=64,
                                        lr = 0.002)

    print('\nepoch_sum:',epoch_sum_predict)



    # 预测模型建模效果
    y_train_pred_0,y_train_pred_1 = model_predict.my_predict(X_predict_train)


    y_test_pred_0,y_test_pred_1 = model_predict.my_predict(X_predict_test)

 



    # 使用NumPy重新构建神经网络架构
    class MyNeuralNetworkNumpy:
        def __init__(self, model, input_size, hidden_size, output_size,ifchuangxin):
            self.ifchuangxin = ifchuangxin
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            # params = {name: param.detach().numpy() for name, param in model.state_dict().items()}
            params = {name: param.detach().cpu().numpy() for name, param in model.state_dict().items()}
            if cengshu == 3:    
                self.weights_fc1 = params['fc1.weight']
                self.bias_fc1 = params['fc1.bias']
                self.weights_fc2 = params['fc2.weight']
                self.bias_fc2 = params['fc2.bias']
                self.weights_fc3 = params['fc3.weight']
                self.bias_fc3 = params['fc3.bias']
                self.weights_fc4 = params['fc4.weight']
                self.bias_fc4 = params['fc4.bias']
            elif cengshu == 2:  
                self.weights_fc1 = params['fc1.weight']
                self.bias_fc1 = params['fc1.bias']
                self.weights_fc2 = params['fc2.weight']
                self.bias_fc2 = params['fc2.bias']
                self.weights_fc3 = params['fc3.weight']
                self.bias_fc3 = params['fc3.bias']
            
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        def relu(self, x):
            return np.maximum(0, x)

        def forward(self, x):
            if cengshu == 3:    
                if self.ifchuangxin:
                    hidden1 = np.dot(x, self.weights_fc1.T) + self.bias_fc1
                    hidden1 = self.relu(hidden1)

                    hidden2 = np.dot(hidden1, self.weights_fc2.T) + self.bias_fc2
                    hidden2 = self.relu(hidden2)

                    hidden3 = np.dot(hidden2, self.weights_fc3.T) + self.bias_fc3
                    hidden3 = self.relu(hidden3)

                    hidden4 = hidden1 + hidden2 + hidden3
                    output = np.dot(hidden4, self.weights_fc4.T) + self.bias_fc4

                else:
                    hidden1 = np.dot(x, self.weights_fc1.T) + self.bias_fc1
                    hidden1 = self.relu(hidden1)

                    hidden2 = np.dot(hidden1, self.weights_fc2.T) + self.bias_fc2
                    hidden2 = self.relu(hidden2)

                    hidden3 = np.dot(hidden2, self.weights_fc3.T) + self.bias_fc3
                    hidden3 = self.relu(hidden3)

                    output = np.dot(hidden3, self.weights_fc4.T) + self.bias_fc4
            elif cengshu == 2:  
                if self.ifchuangxin:
                    hidden1 = np.dot(x, self.weights_fc1.T) + self.bias_fc1
                    hidden1 = self.relu(hidden1)

                    hidden2 = np.dot(hidden1, self.weights_fc2.T) + self.bias_fc2
                    hidden2 = self.relu(hidden2)

                    hidden3 = hidden1 + hidden2
                    output = np.dot(hidden3, self.weights_fc3.T) + self.bias_fc3
                    # hidden2 = self.relu(hidden2)

                    # hidden3 = np.concatenate([x, hidden1, hidden2], axis=1)  # 按列连接
                    # output = np.dot(hidden3, self.weights_fc3.T) + self.bias_fc3

                else:
                    hidden1 = np.dot(x, self.weights_fc1.T) + self.bias_fc1
                    hidden1 = self.relu(hidden1)

                    hidden2 = np.dot(hidden1, self.weights_fc2.T) + self.bias_fc2
                    hidden2 = self.relu(hidden2)

                    output = np.dot(hidden2, self.weights_fc3.T) + self.bias_fc3


                
            return output
        
        
        def my_predict(self, data_input):
            input = data_input  # 随机初始化一个输入序列
            output_prediction = model_numpy.forward(input)
            output_prediction = output_prediction.cpu()
            return output_prediction[:,0], output_prediction[:,1]

    # 使用NumPy模型进行预测
    model_predict.to('cpu') 
    model_numpy = MyNeuralNetworkNumpy(model_predict, input_size, hidden_size, output_size,ischuangxin)
    model_numpy.to(device) 

    y_pred_0, y_pred_1 = model_numpy.my_predict(X_predict_test)

    # 计算 RMSE、MRE
    y_test = y_predict_test
    model_temp = copy.deepcopy(model_predict)
  

    # 的方式公司的r






    # print(last_input_term)
    # print(output_term)
    # print(input_term)


    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.cluster import MeanShift
    from sklearn.decomposition import KernelPCA
    # 提取 output_term 和 input_term 的数据
    def extract_data(data, terms):
        return np.column_stack([data[term] for term in terms])

    # output_data = extract_data(data2_all, output_term+last_output_term)
    # input_data = extract_data(data2_all, input_term)
    output_data = extract_data(data2_all, output_term+last_input_term)
    input_data = extract_data(data2_all, input_term)
    print(output_data.shape)
    print(type(output_data))
    print(f'Output data shape: {output_data.shape}')
    print(f'Input data shape: {input_data.shape}')



    # 使用 KernelPCA 将数据映射到高维空间
    kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.2)
    # transformed_data = kpca.fit_transform(output_data)


    # 使用 K-Means 对 output_term 进行聚类5
    n_clusters = 12
    random_state = ghgjh  # 固定随机数种子
    # kmeans = KMeans(n_clusters=n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    # 使用 K-Means 对 output_term 进行聚类5
    # n_clusters = len(set(labels))
    # # random_state = 42  # 固定随机数种子
    # kmeans = MeanShift()
    labels = kmeans.fit_predict(output_data)

    # n_clusters = len(set(labels))

    # 计算每个子模态的 input_term 空间分布范围
    def compute_limits(input_data, labels, n_clusters):
        limits = {}
        for i in range(n_clusters):
            cluster_data = input_data[labels == i]
            min_limits = cluster_data.min(axis=0)
            max_limits = cluster_data.max(axis=0)
            print(min_limits)
            print(max_limits)
            # limits[i] = {input_term[j]: (min_limits[j], max_limits[j]) for j in range(len(input_term))}
            limits[i] = {input_term[j]: (min_limits[j], max_limits[j]) for j in range(len(input_term))}
        return limits





    input_term_limits = compute_limits(input_data, labels, n_clusters)

    # 输出每个子模态的 input_term 限制范围
    def print_limits(limits):
        for mode, terms in limits.items():
            print(f"Mode {mode}:")
            for term, (min_val, max_val) in terms.items():
                print(f"  {term}: Min={min_val}, Max={max_val}")

    print_limits(input_term_limits)


    



    # 查看 labels 的种类及数量
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    print(f"Number of unique labels: {len(unique_labels)}")
    # 统计每个标签出现的次数
    label_counts = np.bincount(labels)
    # for label, count in enumerate(label_counts):
    #     print(f"Label {label}: {count} points")



    # import numpy as np
    # from sklearn.cluster import KMeans
    # import matplotlib.pyplot as plt
    # from sklearn.decomposition import KernelPCA
    # # 提取 output_term 和 input_term 的数据
    # def extract_data(data, terms):
    #     return np.column_stack([data[term] for term in terms])

    # output_data = extract_data(data2_all, output_term+last_output_term)
    # input_data = extract_data(data2_all, input_term)
    # print(output_data.shape)
    # print(type(output_data))
    # print(f'Output data shape: {output_data.shape}')
    # print(f'Input data shape: {input_data.shape}')



    # # 使用 KernelPCA 将数据映射到高维空间
    # # kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.1)
    # # transformed_data = kpca.fit_transform(output_data)



    # # 使用 K-Means 对 output_term 进行聚类5
    # n_clusters = 5
    # # random_state = 42  # 固定随机数种子
    # kmeans = KMeans(n_clusters=n_clusters)
    # labels = kmeans.fit_predict(output_data)

    # # 计算每个子模态的 input_term 空间分布范围
    # def compute_limits(input_data, labels, n_clusters):
    #     limits = {}
    #     for i in range(n_clusters):
    #         cluster_data = input_data[labels == i]
    #         min_limits = cluster_data.min(axis=0)
    #         max_limits = cluster_data.max(axis=0)
    #         print(min_limits)
    #         print(max_limits)
    #         limits[i] = {input_term[j]: (min_limits[j], max_limits[j]) for j in range(len(input_term))}
    #     return limits





    # input_term_limits = compute_limits(input_data, labels, n_clusters)

    # # 输出每个子模态的 input_term 限制范围
    # def print_limits(limits):
    #     for mode, terms in limits.items():
    #         print(f"Mode {mode}:")
    #         for term, (min_val, max_val) in terms.items():
    #             print(f"  {term}: Min={min_val}, Max={max_val}")

    # print_limits(input_term_limits)


    # def plot_clusters_pairwise(output_data, labels, terms):
    #     plt.figure(figsize=(12, 12))
    #     n_clusters = len(set(labels))  # 假设labels是从0开始编号的簇标签
    #     num_terms = len(terms)

    #     # 遍历所有的维度组合（两两配对）
    #     for i in range(num_terms):
    #         for j in range(i+1, num_terms):
    #             plt.subplot(num_terms-1, num_terms-1, i*(num_terms-1) + j)
    #             for k in range(n_clusters):
    #                 cluster_data = output_data[labels == k]
    #                 plt.scatter(cluster_data[:, i], cluster_data[:, j], s=5, label=f'Mode {k}')
    #             plt.title(f'{terms[i]} vs {terms[j]} space', fontproperties=font)
    #             plt.xlabel(terms[i], fontproperties=font)
    #             plt.ylabel(terms[j], fontproperties=font)
    #             plt.legend()
    #             plt.grid(True)

    #     plt.tight_layout()
    #     plt.show()

    # # 假设output_term和last_output_term组合在一起
    # terms = output_term + last_output_term
    # plot_clusters_pairwise(output_data, labels, terms)



    import matplotlib.pyplot as plt
    import numpy as np

    

    # 获取 input_term 的数量
    # num_terms = len(input_term)
    num_terms = len(input_term)




    iscontrol = True
    # iscontrol = False
    Times = 20*25*80

    Times = 600

    # 过度系数  0.1 越小过度越快
    rou = 0.1
    if_add_noise = 0
    if_gaolu_is_predict = 0
    if_update_model = True
    maxlen = 50

    ifGAN = False

    scalers_X = scalers
    # scalers_X = scalers_X



    # 生成参考轨迹
    def get_yr(aim_value,current_value,alpha,P):
        # 生成设定信号
        setpoint_signal = np.full(10, aim_value)
        # 初始化参数
        alpha = alpha
        y_r = np.zeros(P)
        y_r[0] = current_value
        # 模拟一阶模型
        for k in range(1,P):
            y_r[k] = alpha * y_r[k-1] + (1 - alpha) * aim_value

        # # 绘制结果
        # plt.plot(setpoint_signal, label='Setpoint Signal')
        # plt.plot(y_r,'o-', label='Output Signal (Tracked)')
        # plt.legend()
        # plt.xlabel('Time')
        # plt.ylabel('Amplitude')
        # plt.title('Tracking Setpoint Signal with One-Order Model')
        # plt.show()
        return y_r
    # 测试
    y_r = get_yr(1,-0.5,rou,20)



    20*40*80/278/20




    y_gaolu_train.shape




    y_repeated = np.repeat(y_gaolu_test[10:40], repeats=20, axis=0)

    



    # # 生成期望数据

    def generate_y_aim_data(Times):



        # # 打印原始形状
        # print("Original shape:", y_predict_test[:10].shape)

        # 重复每个数字10次
        y_repeated = np.repeat(y_gaolu_train[65:95], repeats=20, axis=0)
        # y_repeated = np.repeat(y_gaolu_train[195:225], repeats=20, axis=0)
        # y_repeated = np.repeat(y_gaolu_train[0:278], repeats=20, axis=0)

        # y_repeated = np.repeat(y_gaolu_test[10:40], repeats=20, axis=0)

        # # 打印新形状和内容
        # print("New shape:", y_repeated.shape)
        # print(y_repeated)
        if Times == 200:
            set_y1 = np.repeat(np.arange(1455, 1560, 5), 20)[10+75:410-125]
            set_y2 = np.repeat(np.arange(0.34, 0.76, 0.02), 20)[0+75:400-125]



        elif Times == 400:
            set_y1 = np.repeat(np.arange(1455, 1560, 5), 20)[10:410]
            set_y2 = np.repeat(np.arange(0.34, 0.76, 0.02), 20)[0:400]
            
        elif Times == 600:
            # y_repeated = np.repeat(y_gaolu_test[10:40], repeats=20, axis=0)
            set_y1 = y_repeated[:,0]
            set_y2 = y_repeated[:,1]
        elif Times == 278*20:
            
            y_repeated = np.repeat(y_gaolu_train[0:278], repeats=20, axis=0)
            set_y1 = y_repeated[:,0]
            set_y2 = y_repeated[:,1]

        elif Times == 3000:
            y_repeated = np.repeat(y_gaolu_train[120:270], repeats=20, axis=0)
            set_y1 = y_repeated[:,0]
            set_y2 = y_repeated[:,1]
        elif Times == 1000:
            y_repeated = np.repeat(y_gaolu_train[330:380], repeats=20, axis=0)
            set_y1 = y_repeated[:,0]
            set_y2 = y_repeated[:,1]

        # elif Times == 1000:
            # set_y1 = np.repeat(np.arange(1455, 1560, 5), 20)[10:410]
            # set_y2 = np.repeat(np.arange(0.34, 0.76, 0.02), 20)[0:400]
            # set_y1 = np.repeat(np.arange(1457.5, 1562.5, 5), 20)[10:410]
            # set_y2 = np.repeat(np.arange(0.35, 0.77, 0.02), 20)[0:400]




        elif Times == 20*25*80:
            
            # set_y1: 从 1485 到 1534 的整数，每个数字重复 20*35 次
            set_y1 = np.repeat(np.arange(1480, 1560,1), 25 * 20)

            # set_y2: 从 0.30 到 0.64 的小数（步长为 0.01），每个数字重复 20 次，再将整个数组重复 50 次
            set_y2 = np.repeat(np.arange(0.25, 0.50, 0.01), 20)
            set_y2 = np.tile(set_y2, 80)

        else:
            set_y1 = np.full(Times,1500)
            set_y2 = np.full(Times,0.45)

        





        # set_y1_trans = scalers[output_term[0]].transform(set_y1.reshape(-1,1)).flatten()
        # set_y2_trans = scalers[output_term[1]].transform(set_y2.reshape(-1,1)).flatten()

        set_y1_trans = set_y1
        set_y2_trans = set_y2
        set_y1 = scalers[output_term[0]].inverse_transform(set_y1_trans.reshape(-1,1)).flatten()
        set_y2 = scalers[output_term[1]].inverse_transform(set_y2_trans.reshape(-1,1)).flatten()

        return set_y1, set_y2, set_y1_trans, set_y2_trans
    set_y1, set_y2, set_y1_trans, set_y2_trans = generate_y_aim_data(Times)
    # print(set_y1.shape)
    # print(set_y2.shape)
    # print(set_y1_trans.shape)
    # print(set_y2_trans.shape)
    # print(set_y1)
    # print(set_y2)
    # print(set_y1_trans)
    # print(set_y2_trans)





    #生成控制时域的数据格式
    def generate_k_data(u1_data, u2_data, u3_data, u4_data, y1_data,y2_data, num_samples, P):
        nearest_index = np.abs(y1_data - (-0.5)).argmin()
        # 生成随机索引值
        #从原有数据的randint时刻开始往下进行控制
        randint = np.random.randint(1, num_samples - 2 - P - 1)
        randint = nearest_index  # 如果你希望使用固定的值而不是随机生成
        # randint = 250  # 如果你希望使用固定的值而不是随机生成
        print(randint)
        # 提取数据并构成 k_data
        # 第一次得到下面五个变量，固定好格式构成k_data
        u1   = u1_data[randint  :randint+3  ]
        u2   = u2_data[randint  :randint+3  ]
        u3   = u3_data[randint  :randint+3  ]
        u4   = u4_data[randint  :randint+3  ]

        y1   = y1_data[randint  :randint+3  ]
        y2   = y2_data[randint  :randint+3  ]
        k_data = np.concatenate((u1, u2, u3, u4, y1, y2), axis=0)
        print(k_data.shape)

        k_data = np.zeros_like(k_data)
        return k_data




    # # 生成高斯噪声,设置随机种子，以便结果可重现
    # np.random.seed(42)
    # gaussian_noise_SI = np.random.normal(0,d_yuansu*0.001,Times)
    # gaussian_noise_TEMP = np.random.normal(0,d_temp*0.1,Times)
    # # plt.subplot(2, 1, 1)
    # # plt.plot(gaussian_noise_SI)
    # # plt.subplot(2, 1, 2)
    # # plt.plot(gaussian_noise_TEMP)



    def update_model(model_predict,model_gaolu,x,gaolu_data_past_x,gaolu_data_past_y):
        y1_pred, y2_pred = model_gaolu.my_predict(x)
        y_label = np.column_stack((y1_pred, y2_pred))
        gaolu_data_past_x.append(x)
        gaolu_data_past_y.append(y_label)

        
        X_modified = np.array(gaolu_data_past_x)
        y_modified = np.array(gaolu_data_past_y)
        # print(X_modified)
        # print(y_modified)
        X = X_modified.reshape(X_modified.shape[0],X_modified.shape[2])
        Y = y_modified.reshape(y_modified.shape[0],y_modified.shape[2])
        # print(X_modified.shape)
        # print(y_modified.shape)
        if y_modified.shape[0]% 1 == 0:
            model_predict.model_update(X, Y,
                                    epochs=10, 
                                    batch_size=64,
                                    lr = 0.002)
            # gaolu_data_past_x = []
            # gaolu_data_past_y = []

        return model_predict,model_gaolu,gaolu_data_past_x,gaolu_data_past_y



    # 定义单时刻的MPC问题优化
    def my_MPC(k_data,params,M,P,y1_aim,y2_aim,isprint,ifGAN):

        h1 = 1.0
        h2 = 1.0
        lamda1 = 0.01
        lamda2 =lamda1
        lamda3 =lamda1
        lamda4 =lamda1
        y1_percent = 1.0
        y2_percent = 1.0

        # 从固定格式k_data里面读取信息
        u1   = k_data[0:3]
        u2   = k_data[3:6]
        u3   = k_data[6:9]
        u4   = k_data[9:12]

        y1   = k_data[12:15]
        y2   = k_data[15:18]

        
        # 获取猜测值[h U1 U2]

        # h, U1, U2  =params[0], params[1:M+1],params[M+1:]
        if ifGAN:
            params = series2U_numpy_in_control(params,M, generated_numpy,
                                        scalers_X, scalers,
                                        input_term, isprint=False)
        U1, U2, U3, U4  =params[0:M], params[M:2*M],params[2*M:3*M], params[3*M:4*M]
        
        # 整理数据见   MPC推到.escel
        u1   = np.concatenate((u1,U1,U1[-1]*np.ones(P-M)))
        u2   = np.concatenate((u2,U2,U2[-1]*np.ones(P-M)))
        u3   = np.concatenate((u3,U3,U3[-1]*np.ones(P-M)))
        u4   = np.concatenate((u4,U4,U4[-1]*np.ones(P-M)))
        y1   = np.concatenate((y1,np.zeros(P)))
        y2   = np.concatenate((y2,np.zeros(P)))
        if isprint:
            print(u1.round(4))
            print(u2.round(4))
            print(u3.round(4))
            print(u4.round(4))
            print(y1.round(4))    
            print(y2.round(4))
            print('开始预测')

        y1_k = y1[2]
        y2_k = y2[2]





        # 总共预测 P+1 次
        # 对k时刻进行预测-----1次
        for j in range(1):   # j = 0
            x = np.column_stack((   u1[j+2],u2[j+2],u3[j+2],u4[j+2],
                                    u1[j+1],u2[j+1],u3[j+1],u4[j+1],
                                    y1[j+1],y2[j+1]))
            # x = x.reshape((x.shape[0], 1, x.shape[1]))
            y1_m_k, y2_m_k = model_numpy.my_predict(x)
            E1_k = y1_k - y1_m_k
            E2_k = y2_k - y2_m_k
            if isprint:
                print(j,'mode = 0')
                print(x.round(4))
                print(y1_k.round(4),y2_k.round(4))
                print(y1_m_k.round(4),y2_m_k.round(4))

        # 对控制时刻进行预测-----M次
        for j in range(1,M+1):  # j = 1,2
            x = np.column_stack((   u1[j+2],u2[j+2],u3[j+2],u4[j+2],
                                    u1[j+1],u2[j+1],u3[j+1],u4[j+1],
                                    y1[j+1],y2[j+1]))
            # x = x.reshape((x.shape[0], 1, x.shape[1]))
            y1_k_j, y2_k_j = model_numpy.my_predict(x)
            y1[j+2] = y1_k_j.item()
            y2[j+2] = y2_k_j.item()
            if isprint:
                print(j,'mode = 1')
                print(x.round(4))
                print(y1_k_j.round(4),y2_k_j.round(4))
                print('更新后:')
                print(u1.round(4))
                print(u2.round(4))
                print(u3.round(4))
                print(u4.round(4))
                print(u5.round(4))
                print(u6.round(4))
                print(u7.round(4))
                print(y1.round(4))    
                print(y2.round(4))

        # 对控制时域外的部分进行预测-----P-M次
        # 注意：这部分的信号是保持控制不变下进行
        for j in range(M+1,P+1):  #j = 3,4
            x = np.column_stack((   u1[j+2],u2[j+2],u3[j+2],u4[j+2],
                                    u1[j+1],u2[j+1],u3[j+1],u4[j+1],
                                    y1[j+1],y2[j+1]))
            # x = x.reshape((x.shape[0], 1, x.shape[1]))
            y1_k_j, y2_k_j = model_numpy.my_predict(x)
            y1[j+2] = y1_k_j.item()#将预测值作为下一步的输出值
            y2[j+2] = y2_k_j.item()
            if isprint:
                print(j,'mode = 2')
                print(x.round(4))
                print(y1_k_j.round(4),y2_k_j.round(4))
                print('更新后:')
                print(u1.round(4))
                print(u2.round(4))
                print(u3.round(4))
                print(u4.round(4))
                print(y1.round(4))    
                print(y2.round(4))



        k_data2 = np.concatenate((u1[1:4],u2[1:4],u3[1:4],u4[1:4],y1[1:4],y2[1:4]),axis=0)
        if isprint:
            print('更新k_data')
            print(k_data2.round(4))


        #获取参考轨迹
        # 一定要对照好做差的序列
        y1_r_aim  = get_yr(y1_aim,y1_k,rou,P+1)
        y1_r = y1_r_aim[1:] 


        y2_r_aim  = get_yr(y2_aim,y2_k,rou,P+1)
        y2_r = y2_r_aim[1:] 

        y1_M_k = y1[3:]
        y2_M_k = y2[3:]

        # 计算mse
        # lamda1太大的话会导致y1_r和y1_M_k的误差加大*****************导致超调的原因\与目标值之间存在间隙


        y1_err = y1_percent*np.sum((y1_r-(y1_M_k+h1*E1_k))**2) 
        y2_err = y2_percent*np.sum((y2_r-(y2_M_k+h2*E2_k))**2) 
        u1_power = lamda1*np.sum((np.diff(u1[2:]))**2)
        u2_power = lamda2*np.sum((np.diff(u2[2:]))**2)
        u3_power = lamda3*np.sum((np.diff(u3[2:]))**2)
        u4_power = lamda4*np.sum((np.diff(u4[2:]))**2)

        # y1_err = y1_percent*np.sum(np.fabs(y1_r-(y1_M_k+h1*E1_k))) 
        # y2_err = y2_percent*np.sum(np.fabs(y2_r-(y2_M_k+h2*E2_k))) 
        # u1_power = lamda1*np.sum((np.fabs(np.diff(u1))))
        # u2_power = lamda2*np.sum((np.fabs(np.diff(u2))))
        # u3_power = lamda3*np.sum((np.fabs(np.diff(u3))))
        # u4_power = lamda4*np.sum((np.fabs(np.diff(u4))))
        # u5_power = lamda2*np.sum((np.fabs(np.diff(u5))))
        # u6_power = lamda3*np.sum((np.fabs(np.diff(u6))))
        # u7_power = lamda4*np.sum((np.fabs(np.diff(u7))))

        mse = (0
                +y1_err
                +y2_err
                +u1_power
                +u2_power
                +u3_power
                +u4_power
                )
        
        # print('mse {:.7f}'.format(mse))
        if isprint==1:
            print('mse {:.7f}'.format(mse))
            print('1111 {:.7f}'.format(y1_err))
            print('2222 {:.7f}'.format(y2_err))
            print('1111 {:.7f}'.format(u1_power))
            print('2222 {:.7f}'.format(u2_power))
            print('3333 {:.7f}'.format(u3_power))
            print('4444 {:.7f}'.format(u4_power))



        return mse , k_data2, E1_k*h1,  E2_k*h2
        # return mse , k_data2, E1_k*h1



    from shapely.geometry import Polygon, box
    from shapely.geometry import Point



    model_predict = copy.deepcopy(model_temp).to(device) 
    # for name, params in model_predict.named_parameters():
        # print(name, params.data.numpy())



    # # 选择要绘制的多边形
    # key = ('冷风流量', '热风温度')

    # # 确保 expanded_hulls_data 中的值是 Polygon 对象的列表
    # polygons = expanded_hulls_data[key]

    # # 创建正方形边界（范围 [-1, 1]）
    # boundary_box = box(0.5, 0.5, 1, 1)

    # # 创建裁剪后的多边形
    # clipped_polygons = [polygon.intersection(boundary_box) for polygon in polygons if not polygon.intersection(boundary_box).is_empty]

    # x_hull, y_hull = polygons[0].exterior.xy
    # plt.fill(x_hull, y_hull, alpha=0.3, edgecolor='black')
    # x_hull, y_hull = clipped_polygons[0].exterior.xy
    # plt.fill(x_hull, y_hull, alpha=0.3, edgecolor='black')
    # plt.xlim([-1,1])
    # plt.ylim([-1,1])



    penalty_weight = 1.0



    if_zimotai = True


    # 对未来Times周期预测控制
    max_control = 1.0
    # 期望设定值
    set_y1, set_y2, set_y1_trans, set_y2_trans = generate_y_aim_data(Times)

    # MPC参数
    P = 3  # 预测时域长度  3
    M = 3  # 4
    #生成控制时域的数据格式
    k_data = generate_k_data(u1_data, u2_data, u3_data, u4_data,
                            y1_data, y2_data, num_samples, P)

    model_predict = copy.deepcopy(model_temp).to(device) 

    # MPC控制循环   迭代的只有：k_data
    all_pred_y1 = []
    all_pred_y2 = []
    all_pred_u1 = []
    all_pred_u2 = []
    all_pred_u3 = []
    all_pred_u4 = []
    all_state = []
    # 初始化一个最大长度为10的deque
    gaolu_data_past_x = deque(maxlen=maxlen)
    gaolu_data_past_y = deque(maxlen=maxlen)




    print(ghgjh)


    # MPC控制循环4010
    for k in range(Times):
        if iscontrol == False:
            break
        print(f"{ghgjh}这是对第{k}时刻的最优U1、U2输入求解")


        # 定义优化目标函数
        def objective_function(params, *k_data):
            mse, k_data2, E1_k_0, E2_k_0 = my_MPC(k_data=k_data[0], params=params, 
                                    M=M, P=P, 
                                    y1_aim = set_y1_trans[k], y2_aim = set_y2_trans[k],
                                    isprint = 0,ifGAN = ifGAN) 
            return mse 
            # return mse



        # 初始猜测值[h U1 U2]   定义参数的上下限    设置退出条件
        if ifGAN:
            params = np.random.randn(z_dim * M)
            bounds = [(-max_control, max_control) for _ in range(z_dim * M)]
        else:
            params = np.concatenate([np.ones(M)*0.9, np.ones(M)*0.9,np.ones(M)*0.9, np.ones(M)*0.9])
            # print(params)


            if if_zimotai:
                # 加入子模态约束
                if k == 0:
                    new_data = np.column_stack([set_y1_trans[k],set_y2_trans[k],0,0,0,0])
                else:
                    new_data = np.column_stack([set_y1_trans[k],set_y2_trans[k],all_pred_u1[-1],all_pred_u2[-1],all_pred_u3[-1],all_pred_u4[-1]])
                # print(new_data.shape)
                # print(type(new_data))
                
                label = kmeans.predict(new_data)

                # print(label)
                all_state.append(label)
                # print(input_term_limits[label[0]])
                # print(input_term_limits[label[0]][input_term[0]][0])
                # print(input_term_limits[label[0]][input_term[1]])
                # print(input_term_limits[label[0]][input_term[2]])
                # print(input_term_limits[label[0]][input_term[3]])



                bounds =    [(input_term_limits[label[0]][input_term[0]][0], input_term_limits[label[0]][input_term[0]][1]) for _ in range(M)] + \
                            [(input_term_limits[label[0]][input_term[1]][0], input_term_limits[label[0]][input_term[1]][1]) for _ in range(M)] + \
                            [(input_term_limits[label[0]][input_term[2]][0], input_term_limits[label[0]][input_term[2]][1]) for _ in range(M)] + \
                            [(input_term_limits[label[0]][input_term[3]][0], input_term_limits[label[0]][input_term[3]][1]) for _ in range(M)]

            else:
                bounds = [(-max_control, max_control) for _ in range(4 * M)]


            







        options = {
            'maxiter': 1000,      # 最大迭代次数
            # 'disp': True,         # 显示详细的优化过程信息
            'factr': 1e-20,       # 调整收敛精度（降低收敛阈值）
        }

        # 进行优化
        result = minimize(objective_function, 
                        params, 
                        method='COBYLA', 
                        args=k_data,
                        options=options,
                        # constraints=constraints,
                        bounds=bounds)  # 添加约束条件




        if ifGAN:
            result_u = series2U_numpy_in_control(np.array(params),M, generated_numpy,
                                                scalers_X, scalers,
                                                input_term, isprint=False)
            U1, U2, U3, U4 =    result_u[0:M], result_u[M:2*M], \
                                result_u[2*M:3*M], result_u[3*M:4*M]
        else:
            result_u = result.x
            U1, U2, U3, U4 =    result_u[0:M], result_u[M:2*M], \
                                result_u[2*M:3*M], result_u[3*M:4*M]





    # ['富氧流量', '冷风流量', '热风压力', '热风温度']
        u1   = k_data[0:3]
        u2   = k_data[3:6]
        u3   = k_data[6:9]
        u4   = k_data[9:12]

        y1   = k_data[12:15]
        y2   = k_data[15:18]
        u1   = np.concatenate((u1,U1,U1[-1]*np.ones(P-M)))
        u2   = np.concatenate((u2,U2,U2[-1]*np.ones(P-M)))
        u3   = np.concatenate((u3,U3,U3[-1]*np.ones(P-M)))
        u4   = np.concatenate((u4,U4,U4[-1]*np.ones(P-M)))
        y1   = np.concatenate((y1,np.zeros(P)))
        y2   = np.concatenate((y2,np.zeros(P)))




        # for x, y in zip(U2, U4):
        #     point = Point(x, y)
        #     if any(point.within(poly) for poly in clipped_polygons):
        #         print(f'Point ({x}, {y}) is within the polygons.')
        #     else:
        #         print(f'Point ({x}, {y}) is outside the polygons.')




        # 将控制序列第一个数作用于高炉
        j = 1
        x = np.column_stack((   u1[j+2],u2[j+2],u3[j+2],u4[j+2],
                                u1[j+1],u2[j+1],u3[j+1],u4[j+1],
                                y1[j+1],y2[j+1]))
        # x = x.reshape((x.shape[0], 1, x.shape[1]))
        y1_pred0, y2_pred0 = model_predict.my_predict(x)
        if if_gaolu_is_predict:
            y1_pred, y2_pred = model_predict.my_predict(x)
            if if_add_noise:
                # y1_pred = y1_pred+gaussian_noise_TEMP[k].item()
                # y2_pred = y2_pred+gaussian_noise_SI[k].item()
                y1_pred = y1_pred
                y2_pred = y2_pred
        else:
            y1_pred, y2_pred = model_gaolu.my_predict(x)
            if if_update_model:
                # if (np.fabs(y2_pred-y2_pred0)<1.5*d_yuansu) & (np.fabs(y1_pred-y1_pred0)<3*d_temp):
                #     print('sdgsdegerwh')
                
                model_predict.to(device) 
                model_predict,model_gaolu,gaolu_data_past_x,gaolu_data_past_y = update_model(model_predict,model_gaolu,x,gaolu_data_past_x,gaolu_data_past_y)
                # 使用NumPy模型进行预测
                model_predict.to('cpu') 
                model_numpy = MyNeuralNetworkNumpy(model_predict, input_size, hidden_size, output_size,ischuangxin)



        # # 更新k_data
        
        if ifGAN:
            params = result.x
        else:
            params = np.concatenate((U1, U2, U3, U4),axis=0)


        mse, k_data2, E1_k_0, E2_k_0 =my_MPC(k_data=k_data,params=np.array(params),
                                M=M,P=P, 
                                y1_aim = set_y1_trans[k], y2_aim = set_y2_trans[k],
                                isprint = 0,ifGAN = ifGAN)






        all_pred_y1.append(y1_pred)
        all_pred_y2.append(y2_pred)
        all_pred_u1.append(U1[0])
        all_pred_u2.append(U2[0])
        all_pred_u3.append(U3[0])
        all_pred_u4.append(U4[0])
        k_data2[14] = y1_pred.item()
        k_data2[17] = y2_pred.item()
        k_data = k_data2
        # 进入下一时刻，更新预测时域、控制时域，即k_data






    np.max(np.array(all_pred_u1+all_pred_u2+all_pred_u3+all_pred_u4))








    startt = 50
    endd = startt+300



    font222 = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10)  # 替换为你的中文字体文件路径

    y1_pred_inverse_transform = scalers[output_term[0]].inverse_transform(np.array(all_pred_y1[startt:endd]).reshape(-1, 1)).flatten()
    y2_pred_inverse_transform = scalers[output_term[1]].inverse_transform(np.array(all_pred_y2[startt:endd]).reshape(-1, 1)).flatten()
    all_pred_u1_inverse_transform = scalers[input_term[0]].inverse_transform(np.array(all_pred_u1[startt:endd]).reshape(-1, 1)).flatten()
    all_pred_u2_inverse_transform = scalers[input_term[1]].inverse_transform(np.array(all_pred_u2[startt:endd]).reshape(-1, 1)).flatten()
    all_pred_u3_inverse_transform = scalers[input_term[2]].inverse_transform(np.array(all_pred_u3[startt:endd]).reshape(-1, 1)).flatten()
    all_pred_u4_inverse_transform = scalers[input_term[3]].inverse_transform(np.array(all_pred_u4[startt:endd]).reshape(-1, 1)).flatten()


    a1 = scalers[input_term[0]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a2 = scalers[input_term[1]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a3 = scalers[input_term[2]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    a4 = scalers[input_term[3]].inverse_transform(np.array([1,-1]).reshape(-1, 1)).flatten()
    # print(f'上线分别是：{a1}、{a2}、{a3}、{a4}')



    

    rmse_1 = np.mean(np.fabs(set_y1[startt:endd]-y1_pred_inverse_transform))
    rmse_2 = np.mean(np.fabs(set_y2[startt:endd]-y2_pred_inverse_transform))
    print('平均误差',rmse_1.round(4))
    print('平均误差',rmse_2.round(4))


    all_rand_num.append(ghgjh)
    all_pred_y1_mse.append(rmse_1)
    all_pred_y2_mse.append(rmse_2)
    # print('ghghghghhgh',all_rand_num,all_pred_y1_mse,all_pred_y1_mse)

    # 将数据保存到字典中
    datasssss = {
        "all_rand_num": all_rand_num,
        "all_pred_y1_mse": all_pred_y1_mse,
        "all_pred_y2_mse": all_pred_y2_mse
    }

    # 将字典转换为 DataFrame
    datasssss_df = pd.DataFrame(datasssss)

    # 保存到 Excel 文件
    datasssss_df.to_excel("output.xlsx", index=False)

