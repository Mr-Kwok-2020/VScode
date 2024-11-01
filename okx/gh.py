import okx.MarketData as MarketData
import tkinter as tk
from threading import Thread, Event
import time
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import textwrap

# 初始化API
flag = "0"  # 实盘:0 , 模拟盘：1
marketDataAPI = MarketData.MarketAPI(flag=flag)
instID = ['BTC', 'DOGE', 'PEPE']

# 创建主窗口
root = tk.Tk()
main_x = 125
main_y = 96
root.geometry(f"{main_x}x{main_y}+{0}+{int(1440-main_y)}")
# main_x = 500
# main_y = 300
# root.geometry(f"{main_x}x{main_y}+{500}+{int(500)}")
root.attributes('-topmost', True)
root.attributes('-transparentcolor', 'yellow')
root.config(bg='yellow')
root.overrideredirect(True)

# label = tk.Label(root, text="", font=("Helvetica", 9), bg='red', anchor='w', justify='left')
label = tk.Label(root, text="", font=("Helvetica Neue Regular", 12), bg='yellow', anchor='w', justify='left')

label.pack(pady=1, padx=1, fill='both', expand=True)

# 全局变量
display_text = ""
heart = 0  # 状态位初始为 "0"
is_dragging = False  # 是否拖动的标志
last_result_processes = {}
last_gray_level = (0, 0, 0)
start_x = 0
start_y = 0

# 矩形区域的设置
rect_win = None  # 矩形窗口的引用
canvas =  None  
# 线程停止事件
stop_event = Event()

# # 创建 Treeview 小部件
# tree = ttk.Treeview(root, columns=('Ticker', 'Price', 'Heart'), show='headings')
# tree.heading('Ticker', text='Ticker')
# tree.heading('Price', text='Price')
# tree.heading('Heart', text='Heart')
# tree.pack(pady=1, padx=1, fill='both', expand=True)



def get_average_color(image):
    """计算图像的平均颜色"""
    np_image = np.array(image)
    # 确保图像数据有效
    if np_image.size == 0:
        return (0, 0, 0)
    
    # 计算每个像素的平均值（RGB）
    avg_color = np_image[:, :, :3].mean(axis=(0, 1))
    # print(avg_color)
    return tuple(map(int, avg_color))

def calculate_remaining_area_avg_color(x, y, width, height, expand_ratio=0.1):
    """
    计算屏幕上截取区域去除部分后的剩余区域的平均颜色。
    
    参数:
        x, y: 矩形区域的左上角坐标。
        width, height: 矩形区域的宽度和高度。
        expand_ratio: 截图区域扩展的比例（默认为1，即扩大100%）。
    
    返回:
        剩余区域的平均颜色，返回RGB格式的三元组。
    """
    
    # # 扩大截图区域
    # expanded_width = int(width * (1 + expand_ratio))
    
    # # 截取屏幕区域
    # image = ImageGrab.grab(bbox=(x, y, x + expanded_width,  y + height))
    # # print('1', x, y, x + expanded_width, height)

    # # 计算裁剪掉区域的坐标
    # crop_x1 = width
    # crop_y1 = 0
    # crop_x2 = expanded_width
    # crop_y2 = height
    
    # cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    # # print('2', crop_x1, crop_y1, crop_x2, crop_y2)

    # # 计算剩余区域的平均颜色
    # avg_color = get_average_color(cropped_image)

    image = ImageGrab.grab(bbox=(x, y, x + width,  y + height))






    # 计算剩余区域的平均颜色
    avg_color = get_average_color(image)
    
    return avg_color

def calculate_luminance(rgb_color):
    global last_gray_level
    """计算RGB颜色的感知亮度"""
    r, g, b = rgb_color
    gray_level = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    print(f"当前灰度值: {gray_level}")

    # 检查灰度值是否在0.45到0.55之间
    if 0.45 <= gray_level <= 0.55:
        # 如果在范围内，返回上一次的结果
        return last_gray_level
    else:
        # 根据灰度值更新last_gray_level
        last_gray_level = (0, 0, 0) if gray_level > 0.5 else (255, 255, 255)
        return last_gray_level

def update_label_background():
    global label, rect_win

    # 获取矩形窗口的区域
    x = root.winfo_rootx()
    y = root.winfo_rooty()
    width = root.winfo_width()
    height = root.winfo_height()

    avg_color = calculate_remaining_area_avg_color(x, y, width, height, expand_ratio = 1.0)

    hex_color = '#%02x%02x%02x' % avg_color    

    contrasting_color = calculate_luminance(avg_color)
    hex_font_color = '#%02x%02x%02x' % contrasting_color
    
    # # 更新Label的背景色和字体颜色
    # label.config(bg=hex_color, fg=hex_font_color)
    label.config(fg=hex_font_color)

    root.after(500, update_label_background)

import httpx
import time

def get_market_data_with_retry(api, max_retries=5, delay=2):
    retries = 0
    while retries < max_retries:
        try:
            # 发起请求
            result = api.get_tickers(instType="SWAP")
            return result
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e}")
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        
        retries += 1
        print(f"Retrying... ({retries}/{max_retries})")
        time.sleep(delay)
    
    print("Failed to retrieve data after several attempts.")
    return None

# 使用重试机制获取数据
result = get_market_data_with_retry(marketDataAPI)

def update_data():
    global display_text, heart, marketDataAPI, last_result_processes
    while not stop_event.is_set():
        result = get_market_data_with_retry(marketDataAPI)
        if result is None:
            continue

        # result = marketDataAPI.get_tickers(instType="SWAP")
        result_processes = {
            symbol: item['last']
            for symbol in instID
            for item in result['data']
            if f"{symbol}-USDT-SWAP" == item['instId']
        }

        result_processes['PEPE'] = f"{float(result_processes['PEPE']) * 1e5:.4f}"
        result_processes[instID[1]] = f"{float(result_processes[instID[1]]):.3f}"
        # 在字典中加入心跳状态
        result_processes['Heart'] = str(heart)

        display_text = "\n".join([f"{symbol:5} : {price:3}" for symbol, price in result_processes.items()])

        root.after(0, lambda: label.config(text=display_text))

        print(result_processes)
        # time.sleep(1)


        if last_result_processes == result_processes:
            heart = heart + 1
        else:
            heart = 0

        last_result_processes = result_processes

        if heart > 5:
                heart = 0
                marketDataAPI = MarketData.MarketAPI(flag=flag)
                print('sdfghjksdfghjklfgio')
        

def on_mouse_enter(event):
    global is_dragging
    if not is_dragging:
        root.config(bg='lightgrey')
        # 创建独立的矩形窗口
        create_rectangle_window()

def on_mouse_leave(event):
    global rect_win, canvas
    root.config(bg='white')
    # 删除创建的独立的矩形窗口及其红色边框
    rect_win.destroy()
    # rect_win = None


def on_left_button_down(event):
    global is_dragging, start_x, start_y
    is_dragging = True
    start_x = event.x
    start_y = event.y

def on_left_button_up(event):
    global is_dragging
    is_dragging = False
    check_if_in_rectangle()

def on_mouse_drag(event):
    if is_dragging:
        x = root.winfo_x() + event.x - start_x
        y = root.winfo_y() + event.y - start_y
        root.geometry(f'+{x}+{y}')
        check_if_in_rectangle()

def check_if_in_rectangle():
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    win_width = root.winfo_width()
    win_height = root.winfo_height()

    rect_x = rect_win.winfo_x()
    rect_y = rect_win.winfo_y()
    rect_win_width = rect_win.winfo_width()
    rect_win_height = rect_win.winfo_height()

    if (win_x + win_width > rect_x) and (win_x < rect_x + rect_win_width) and \
        (win_y + win_height > rect_y) and (win_y < rect_y + rect_win_height):
        stop_program()

def stop_program():
    root.destroy()
    rect_win.destroy()
    stop_event.set()
def create_rectangle_window():
    global rect_win, canvas
    if rect_win is not None:
        rect_win.destroy()

    # 获取主窗口的位置和尺寸
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    win_width = root.winfo_width()
    win_height = root.winfo_height()
    
    # 计算矩形窗口的位置
    rect_x = win_x + win_width
    rect_y = win_y

    rect_width = 1000
    rect_height = 600

    rect_win = tk.Toplevel()
    rect_win.geometry(f"{rect_width}x{rect_height}+{int((root.winfo_screenwidth()-rect_width)/5)}+{int((root.winfo_screenheight()-rect_height)/2)}")
    rect_win.overrideredirect(True)
    rect_win.attributes('-topmost', True)
    rect_win.config(bg='yellow')
    rect_win.wm_attributes('-transparentcolor', 'yellow')# 设置背景透明

    # 绘制红色边框
    canvas = tk.Canvas(rect_win, width=rect_width, height=rect_height, highlightthickness=0)
    canvas.pack()
    # canvas.create_rectangle(0, 0, rect_width, rect_height, outline='red', width=10)
    canvas.create_rectangle(0, 0, rect_width, rect_height, outline='red', width=10, tag="one")
    rect_win.lift()




# # 绑定鼠标事件
# root.bind("<Enter>", on_mouse_enter)
# root.bind("<Leave>", on_mouse_leave)
# root.bind("<Button-1>", on_left_button_down)
# root.bind("<B1-Motion>", on_mouse_drag)
# root.bind("<ButtonRelease-1>", on_left_button_up)

# # 启动背景色更新
# update_label_background()

# 启动更新数据的后台线程
update_thread = Thread(target=update_data, daemon=True)
update_thread.start()
update_background = Thread(target=update_label_background, daemon=True)
update_background.start()

# 启动tkinter主循环
root.mainloop()
