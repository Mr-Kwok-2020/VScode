@echo off  
timeout /t 1 /nobreak  
  
:: 假设pythonw.exe在你的Python环境中可用  
set PYTHONW_PATH=E:\conda\envs\py39_tf\pythonw.exe  
  
:: 启动第一个 Python 脚本  
start "" "%PYTHONW_PATH%" "C:\Users\haokw\Documents\GitHub\self\qidian\qidian_exp_award_task.py"  
  
:: 启动第二个 Python 脚本  
start "" "%PYTHONW_PATH%" "C:\Users\haokw\Documents\GitHub\VScode\okx\gh.py"
