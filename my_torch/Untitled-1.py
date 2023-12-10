# %%
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import os
import warnings
# 禁用警告
warnings.filterwarnings("ignore")
device="cuda"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import gc
import time
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
NUM_WORKERS = 4
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
from datetime import datetime

# %%
# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
end = datetime.now()
start = datetime(end.year -3, end.month-0, end.day)

for stock in tech_list:
    globals()[stock] = yf.download(stock, start, end)

# %%
company_data = [AAPL, GOOG, MSFT, AMZN]
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]
for company_data, com_name in zip(company_data, company_name):
    company_data["company_name"] = com_name
    
# df = pd.concat(company_list, axis=0)
# df.tail(10)

# %%

AAPL_Close_df = AAPL.filter(['Close'])
print(type(AAPL_Close_df))
print(AAPL_Close_df.shape)
AAPL_Close =AAPL_Close_df.values
print(type(AAPL_Close))
print(AAPL_Close.shape)

# %%
# 归一化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
AAPL_Close = scaler.fit_transform(AAPL_Close)

AAPL_Close
print(AAPL_Close.shape)

# %%
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
num_sample = 10
training_data_len = int(np.ceil(len(AAPL_Close)*0.8))
for i in range(num_sample,training_data_len):
    print(i)
    x_train.append(AAPL_Close[i-num_sample:i])
    y_train.append(AAPL_Close[i])
    print(AAPL_Close[i-num_sample:i].shape,AAPL_Close[i])

x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train.shape,y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
print(x_train.shape,y_train.shape,type(x_train),type(y_train))

# # dataset = np.column_stack((data, labels))
# dataset = np.column_stack((x_train, y_train))

# %%
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()  # 输入数据数组，形状为(670, 10, 1)
        self.labels = torch.from_numpy(labels).float()  # 对应标签数组，形状为(670, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
# dataset = MyDataset(data, labels)
train_dataset = MyDataset(x_train, y_train)

print(len(train_dataset))
print(train_dataset[0][0].shape)# 数据   dataset[第几个样本][0 数据  1标签]
print(train_dataset[0][1].shape)# 对应标签

# %%
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim = 1, 256 // 2),
            nn.ReLU(),
            nn.Linear(256 // 2, 256),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=256, 
            batch_first=True, 
            bidirectional=True)
        
        # 添加一个全连接层
        self.logits = nn.Sequential(
            nn.Linear(lstm_dim=256 * 2, logit_dim=64),
            nn.ReLU(),
            nn.Linear(logit_dim=64, num_classes=1),
        )

    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.lstm(features)
        predictions = self.logits(features)

        return predictions


# %%
model = RNNModel()

# %%
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
)

# %%

criterion = torch.nn.MSELoss()

optim = torch.optim.Adam(model.parameters(),lr=0.001)

# %%
epochs = 100
for epoch in range(epochs):
    model.train()
    model.zero_grad()
    avg_loss = 0
    start_time = time.time()
    for id,(data,label) in enumerate(train_loader):
        # print(id,(data,label))
        pred = model(data)

        loss = criterion(pred,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(train_loader)
    elapsed_time = time.time() - start_time
    
    elapsed_time = elapsed_time
    print(
        f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
        f"loss={avg_loss:.8f}")

# %%
lr = 0.001
optimizer = getattr(torch.optim, "Adam")(model.parameters(), lr=lr)

# %%
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=32, epochs=1)

# %%

test_data = AAPL_Close[training_data_len-num_sample:]
# print(test_data)
x_test = []
y_test = []
for i in range(num_sample, len(test_data)):
    # print(i)
    # print(test_data[i-10:i])
    x_test.append(test_data[i-num_sample:i])
    y_test.append(test_data[i])
    
# Convert the data to a numpy array
x_test,y_test = np.array(x_test),np.array(y_test)
# print(x_test)
# print(x_test.shape,y_test.shape)
# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
y_test = np.reshape(y_test, (y_test.shape[0], 1 ))
print(x_test.shape,y_test.shape)
print(x_test)
# Get the models predicted price values 
predictions = model.predict(x_test)
# print(predictions.shape)
print(predictions)
predictions = scaler.inverse_transform(predictions)
print(predictions)
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# %%
# Plot the data
train = AAPL_Close_df[:training_data_len+1]
valid = AAPL_Close_df[training_data_len:]

valid['Predictions'] = predictions
print(valid['Predictions'])
# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(AAPL_Close_df['Close'],'o')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['yuan ','Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



