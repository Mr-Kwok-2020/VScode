import requests

# Clash external controller 的地址
controller_url = 'http://127.0.0.1:9090'

# 修改 DNS enable 字段的函数
def update_dns_enable(enable_value):
    url = f'{controller_url}/configs'
    
    # 发送 PATCH 请求来修改 enable 的值
    response = requests.patch(url, json={"dns": {"enable": enable_value}})
    
    if response.status_code == 204:
        print(f"成功将 DNS enable 修改为 {enable_value}")
    else:
        print(f"修改失败，错误代码: {response.status_code}")
        print(response.text)

# 检查并修改 DNS enable 的值
def check_and_update_dns_enable():
    url = f'{controller_url}/configs'
    
    # 获取当前的配置
    response = requests.get(url)
    
    if response.status_code == 200:
        current_config = response.json()
        dns_enable = current_config.get('dns', {}).get('enable')
        
        # 如果 dns.enable 是 True，修改为 False
        if dns_enable:
            print("DNS enable 当前为 True，正在修改为 False...")
            update_dns_enable(False)
        else:
            print("DNS enable 已经为 False，无需修改。")
    else:
        print(f"获取配置失败，错误代码: {response.status_code}")
        print(response.text)

# 执行检查和修改操作
check_and_update_dns_enable()
