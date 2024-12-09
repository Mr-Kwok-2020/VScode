@echo off  
timeout /t 1

:: 假设pythonw.exe在你的Python环境中可用  
set PYTHONW_PATH=E:\conda\envs\py39_tf\pythonw.exe  
  
:: 启动第一个 Python 脚本  
start "" "%PYTHONW_PATH%" "C:\Users\haokw\Documents\GitHub\VScode\qidian\qidian_web\qidian_exp_award_task.py"  





  


