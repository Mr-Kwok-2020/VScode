@echo off  
timeout /t 1 /nobreak  
  
:: 假设pythonw.exe在你的Python环境中可用  
set PYTHONW_PATH=D:\Software\conda\envs\py39_tf\python.exe  

:: 启动第一个 Python 脚本  
start "" "%PYTHONW_PATH%" "C:\Users\Admin\Documents\GitHub\VScode\okx\实时价格监测.py"  

