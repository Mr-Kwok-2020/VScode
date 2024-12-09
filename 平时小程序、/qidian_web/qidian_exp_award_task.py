
import requests
from bs4 import BeautifulSoup
import re
import json
from lxml import etree
import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service  
import psutil
from selenium.webdriver.common.by import By
import time
from datetime import datetime
import random
import time
import random
from datetime import datetime
import psutil
from selenium import webdriver
from selenium.webdriver.chrome.service import Service  
from selenium.webdriver.common.by import By

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager



class MyStruct:
    def __init__(self, name, number, max):
        self.name = name
        self.number = number
        self.max = max



class TimeUtility:
    def __init__(self):
        pass

    def get_current_timestamp(self):
        """获取当前时间戳（秒）"""
        return int(time.time())

    def get_current_time(self, fmt='%Y-%m-%d %H:%M:%S'):
        """获取格式化的当前时间"""
        return datetime.now().strftime(fmt)


def kill_background_task(task_name):
    # 遍历所有进程
    for proc in psutil.process_iter(['pid', 'name']):
        # 检查进程名称是否匹配
        if task_name in proc.info['name']:
            # 终止进程
            proc.terminate()


def is_specific_page(driver,user_ID):

    time.sleep(10)

    # 选择器和期望的内容
    css_selector = "#headerLog"
    expected_content = user_ID.name
    
    print('验证登陆信息')

    try:
        # 等待并检查特定元素内容
        element = WebDriverWait(driver, 0).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
        )
        if expected_content in element.text:
            print("已登录")
            print(user_ID.name,user_ID.number)
            return True
        else:
            print("未登录")
            return False
    except:
        
        return False


def loginQQ(driver,user_ID):
    print("进行QQ登录")
    # 切换到 iframe
    driver.get("https://graph.qq.com/oauth2.0/show?which=Login&display=pc&g_ut=1&response_type=code&redirect_uri=https%3A%2F%2Fptlogin.qidian.com%2Flogin%2Fqqconnectcallback%3Freturnurl%3Dhttps%253A%252F%252Fmy.qidian.com%26appid%3D10%26areaid%3D1%26jumpdm%3Dqidian%26popup%3D0%26ajaxdm%3D%26target%3Dtop%26ticket%3D0%26ish5%3D%26auto%3D1%26autotime%3D30%26format%3D%26method%3D&client_id=100486335")
    
    # 等待iframe加载完成并切换到该iframe
    WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it((By.ID, "ptlogin_iframe")))
    
    # 使用JavaScript查找目标元素并执行点击操作
    script = f"""
        var element = document.querySelector('a[uin="{user_ID.number}"]');
        element.click();
    """
    driver.execute_script(script)
    time.sleep(5)
    
    driver.get("https://my.qidian.com/level")
    


def count_received_rewards(driver,urls):
    # 获取当前页面的源代码
    driver.get(urls)
    page_source = driver.page_source

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # 查找所有具有 data-num 属性的元素
    elements_with_data_num = soup.find_all(attrs={"data-num": True})

    # 统计已领取的奖励数量
    received_count = 0
    for element in elements_with_data_num:
        if '已领取' in element.get_text():
            received_count += 1

    # 返回已领取的奖励数量
    return received_count


def exp_award_task(driver,urls):
    driver.get(urls)
    # 找到并点击领取按钮
    try:
        # 通过class和其他属性定位领取按钮
        claim_button = driver.find_element(By.CLASS_NAME, "elGetExp")
        claim_button.click()
        print("领取按钮点击成功")
        
    except Exception as e:
        print(f"未能找到或点击领取按钮")



def config(urls):
    user_data_dir = r"C:\Users\haokw\AppData\Local\Google\Chrome\User Data"
    # service = Service(r'C:\Program Files\Google\Chrome\Application\chromedriver.exe')
    service = webdriver.chrome.service.Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-data-dir={user_data_dir}")
    # options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--ignore-certificate-errors")  # 忽略 SSL 错误
    options.add_argument("--log-level=3")  # 设定日志等级
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Your User-Agent")

    kill_background_task("chrome")
    driver = webdriver.Chrome(options=options, service=service)
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
    'source': '''
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        })
    '''
})
    # # 禁用控制台日志的JavaScript代码-
    #     # script = """
    #     (function() {
    #         var originalLog = console.log;
    #         var originalWarn = console.warn;
    #         var originalError = console.error;
    #         console.log = function() {};
    #         console.warn = function() {};
    #         console.error = function() {};
    #     })();
    # """
    driver.get(urls)
    # driver.execute_script(script)
    time.sleep(1)
    return driver



def main():
    urls = "https://my.qidian.com/level"
    driver = config(urls)

    # 创建一个实例
    user_2428392526 = MyStruct(name="书虫小浩浩", number="2428392526",max = 8)
    user_2739390074 = MyStruct(name="书友20230331232311678", number="2739390074",max = 6)
    user_ID = user_2428392526

    # 调用函数检查是否已经登录
    logged_in = is_specific_page(driver,user_ID = user_ID)
    while(logged_in  == False):
        loginQQ(driver,user_ID = user_ID)
        logged_in = is_specific_page(driver,user_ID = user_ID)


    time_util = TimeUtility()


    # 定义在线经验值奖励
    exp_award_task_Cnt = 0  # 确保计数器在函数外部定义，以便在多次调用时保留其值


    print("开始：")
    while True:    
        print('运行时刻：', time_util.get_current_time())
        exp_award_task(driver,urls)

        received_count = count_received_rewards(driver,urls)
        print(f"已完成任务次数: {received_count}/8")

        if received_count == user_ID.max:
            break    

        time.sleep(300 + random.randint(1, 60))


    driver.quit()




if __name__ == "__main__":
    main()
