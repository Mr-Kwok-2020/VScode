// 宏参数定义
var ham = hamibot.env;
var ham = {
    // 投推荐票书名（男/女）
    "bookName_tuijian_nan": "青山",
    // 投推荐票书名（女）
    "bookName_tuijian_nv": "第一侯",
    // 日常任务翻书书名
    "bookName_read": "第一序列",
    // 听书书名
    "listenBookName": "都重生了谁谈恋爱啊丨都市爽文丨搞笑甜宠丨精品多人有声剧",
    // 看视频数量
    "sp": 0,
    // 点击函数中，保存初次传入的控件
    "InitialValue": null,
    // 定义解锁屏幕每个数字的坐标
    "coordinates": {
        1: [250, 1500],
        2: [600, 1500],
        3: [950, 1500],
        4: [250, 1750],
        5: [600, 1750],
        6: [950, 1750],
        7: [250, 2000],
        8: [600, 2000],
        9: [950, 2000],
        0: [600, 2250]
    },
    // 解锁密码 '985211'
    "password": [9, 8, 5, 2, 1, 1],
    // 信息输出窗口
    "bounds": undefined,
    // 输出窗口中心X
    "centerX": undefined,
    // 输出窗口中心Y
    "centerY": undefined,
    // 输出窗口左边界
    "left": undefined,
    // 输出窗口右边界
    "right": undefined,
    // 输出窗口上边界
    "top": undefined,
    // 输出窗口下边界
    "bottom": undefined,

    "isHomeText": ["QQ", "起点读书", "微信"]
};



// 底层点击触摸滑动等工具函数
function jstime(textObj) {
    // 如果 textObj 为 null，则返回 null
    if (textObj == null) {
        return null;
    }

    // 存储初始文本内容
    var initText = textObj.text();

    // 从文本中提取时间（数字）
    var match = initText.match(/\d+/g);

    // 如果找到数字，返回第一个数字的整数值；否则返回 null
    return match ? parseInt(match[0]) : null;
}
function getXy(obj) {
    // 如果 obj 为 null，则返回 null
    if (obj == null) {
        return null;
    }

    // 获取对象的边界信息
    var bounds = obj.bounds();

    // 计算并返回中心坐标
    return {
        centerX: (bounds.left + bounds.right) / 2,
        centerY: (bounds.top + bounds.bottom) / 2
    };
}
function clickCenter(params) {
    // 获取对象的中心坐标
    var center = getXy(params);

    // 如果未找到中心坐标，输出提示信息并返回
    if (center == null) {
        console.log('没找到');
        return;
    }

    // 执行点击操作
    click(center.centerX, center.centerY);
    console.log('点击坐标');
}
function backHome(params) {
    // 循环返回，直到找到 ID 为 "normal" 的元素
    do {
        back();
    } while (id("normal").findOne(500) == null);

    // 进入书架界面
    back_shujia();
    sleep(1000);

    // 输出已到主界面
    log('已到主界面');
}
function clickParentIfClickable(widget) {
    // 保存初始控件
    if (ham.InitialValue == null) {
        ham.InitialValue = widget;
    }

    // 如果 widget 为空，输出提示并终止递归
    if (widget === null) {
        console.log('找不到');
        ham.InitialValue = null;
        return null;
    }

    // 如果点击成功，输出提示并结束递归
    if (widget.click()) {
        console.log('已点击');
        ham.InitialValue = null;
        return true;
    }

    // 获取父控件并进行递归调用
    var parentWidget = widget.parent();
    if (parentWidget === null) {
        console.log('不可点击');
        clickCenter(ham.InitialValue);
        ham.InitialValue = null;
        return false;
    }

    // 递归调用自身，传入父控件进行下一次查找和点击
    return clickParentIfClickable(parentWidget);
}
function longClickParentIfClickable(widget) {
    // 如果 widget 为空，输出提示并终止递归
    if (widget === null) {
        console.log('找不到');
        return null;
    }

    // 如果长按成功，输出提示并结束递归
    if (widget.longClick()) {
        console.log('已长按');
        return true;
    }

    // 获取父控件并进行递归调用
    var parentWidget = widget.parent();
    if (parentWidget === null) {
        console.log('不可长按');
        return false;
    }

    // 递归调用自身，传入父控件进行下一次查找和长按
    return longClickParentIfClickable(parentWidget);
}



// 手机底层工具函数
function keep_silent() {
    // device.setMusicVolume(0) - 将音乐音量设置为 0
    device.setMusicVolume(0);
}
function setScreenBrightness() {
    // 开启自动管理屏幕光照强度
    console.log('开启自动管理屏幕光照强度');

    // 初始化亮度
    var Brightness_value = device.getBrightness();

    // 设置光线传感器监听器
    var sensor = sensors.register('light');
    sensor.on('change', (event, light) => {

        // 如果当前为手动模式，则切换为自动模式
        if (device.getBrightnessMode() == 1) {
            device.setBrightnessMode(0);
        }

        var lightIntensity = Math.floor(light); // 将光强转换为整数
        var min_limit = 6;
        var max_limit = 50;

        // 根据光强计算亮度
        if (lightIntensity < min_limit) {
            Brightness_value = 0;
        } else if (lightIntensity > max_limit) {
            Brightness_value = 225;
        } else {
            // 插值计算亮度
            Brightness_value = Math.floor((lightIntensity - min_limit) / (max_limit - min_limit) * (225 - 0));
        }

        // 设置屏幕亮度
        device.setBrightness(Brightness_value);

        // console.log('光强：', lightIntensity, '设置亮度：', Brightness_value);
    });
}
function get_current_time() {
    var now = new Date();

    var hours = now.getHours(); // 获取当前小时
    var minutes = now.getMinutes(); // 获取当前分钟
    var seconds = now.getSeconds(); // 获取当前秒
    var milliseconds = now.getMilliseconds(); // 获取当前毫秒

    // 输出当前时间
    console.log("当前时间: " + hours + ":" + minutes + ":" + seconds + "." + milliseconds);

    // 返回时间对象
    return {
        hours: hours,
        minutes: minutes,
        seconds: seconds,
        milliseconds: milliseconds
    };
}
function get_timestamp() {
    // 获取当前时间戳
    var timestampInSeconds = Math.floor(new Date().getTime() / 1000);

    // 输出时间戳
    // console.log("当前时间戳: " + timestampInSeconds);

    return timestampInSeconds;
}





// 手机辅助函数

function isHomeScreen() {
    // 是否符合主页面的特征
    if (text(ham.isHomeText[0]).exists() && 
        text(ham.isHomeText[1]).exists() && 
        text(ham.isHomeText[2]).exists()) {
        console.log("主页面");
        return true;
    }
    console.log("不是主页面");
    sleep(2000);
    return false;
}
function unlockScreen() {
    console.log("开始解锁屏幕--------");

    // 如果屏幕已亮则锁屏
    if (device.isScreenOn()) {
        console.log('当前屏幕已亮，先锁屏');
        lockScreen();
    }

    do {
        // 唤醒设备
        device.wakeUpIfNeeded();
        sleep(500);

        // 模拟手势从屏幕底部滑动到屏幕中间
        console.log("开始上滑进入输入密码");
        gesture(200, [ham.centerX, ham.top], [ham.centerX, ham.bottom]);
        sleep(300);

        // 输入密码
        for (var i = 0; i < ham.password.length; i++) {
            var num = ham.password[i];
            var coord = ham.coordinates[num];
            click(coord[0], coord[1]);
            sleep(100); // 短暂等待，确保点击操作有效
        }

        // 返回主屏幕
        home();
        home();
        sleep(500);

    } while (!isHomeScreen());

    console.log("解锁成功");
    sleep(1000);
    home();
    home();
}
function lockScreen() {
    // 返回主屏幕
    home();
    home();

    // 执行锁定屏幕操作
    var success = runtime.accessibilityBridge.getService().performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_LOCK_SCREEN);

    // 可选：检查是否成功锁屏并输出结果
    if (success) {
        console.log('锁屏成功');
    } else {
        console.log('锁屏失败');
    }
}
function checkDateTimeAfterSet(setDay, setHours, setMinutes) {
    // 打印当前时间和日期
    console.log("当前日期:", currentDay);
    console.log("当前时间:", currentHours, "小时", currentMinutes, "分钟");

    // 打印设定时间
    if (dateSet) {
        console.log("设定日期:", dateSet.day);
    }
    if (timeSet) {
        console.log("设定时间:", timeSet.hours, "小时", timeSet.minutes, "分钟");
    }



    // 获取当前日期
    var now = new Date();
    // 获取当前日期的天数
    var day = now.getDate();
    log('当前日期为：', day)

    // 检查当前日期是否大于2号
    if (day >= setDay) {
        return true;
    }

    var currentTime = get_current_time();
    if (currentTime.hours >= setHours && currentTime.minutes >= setMinutes) {
        return true;
    }
    return false;



}
function checkDateTimeAfterSet(setDay, setHours, setMinutes) {
    // 获取当前时间
    var now = new Date();
    var currentDay = now.getDate();
    var currentTime = get_current_time();
    var currentHours = currentTime.hours;
    var currentMinutes = currentTime.minutes;

    // 打印当前时间和日期
    console.log("当前日期:", currentDay);
    console.log("当前时间:", currentHours, "小时", currentMinutes, "分钟");

    // 打印设定时间
    console.log("设定日期:", setDay);
    console.log("设定时间:", setHours, "小时", setMinutes, "分钟");

    // 检查当前日期是否超过设定日期
    if (currentDay > setDay) {
        return true;
    }
    else if (currentTime.hours >= 23 && currentTime.minutes >= 40) {
        return true;
    }

    // 如果日期和时间都未超过设定值，则返回 false
    return false;
}

























// 基本任务函数
function configure() {
    console.log("开始初始化");

    if (auto.service == null) {
        console.log("请先开启无障碍服务！");
    } else {
        console.log("无障碍服务已开启");

        // 等待一段时间以确保操作稳定
        sleep(200);

        // 显示控制台
        console.hide();
        console.show();
        console.log("屏幕尺寸宽、高:", device.width, device.height);

        // 等待无障碍服务启动
        auto.waitFor();

        // 配置控制台
        console.setTitle("起点自动任务");
        console.setPosition(100, 300);
        console.setSize(800, 1300);

        // 获取并设置元素的边界和中心坐标
        var bounds = className("android.widget.FrameLayout").depth(0).findOne();

        var xy = getXy(bounds);
        ham.centerX = xy.centerX;
        ham.centerY = xy.centerY;

        ham.bounds = bounds
        ham.right = bounds.bounds().right;
        ham.left = bounds.bounds().left;
        ham.top = bounds.bounds().top;
        ham.bottom = bounds.bounds().bottom;

        console.log("元素的左边界: " + ham.left);
        console.log("元素的上边界: " + ham.top);
        console.log("元素的右边界: " + ham.right);
        console.log("元素的下边界: " + ham.bottom);
        console.log("元素的中心X: " + ham.centerX);
        console.log("元素的中心Y: " + ham.centerY);

        console.log("初始化结束");
    }
}






// 任务线






// 多线程


configure()

var thread1 = threads.start(function () {
    setScreenBrightness();
    while (1) {
        sleep(1000)
        keep_silent()
    }
});





// 测试样本
function testCheckDateTimeAfterSet() {
    console.log("测试样本结果:");

    // 1. 测试当前日期和时间是否超过8月2号且时间为23:50
    var result1 = checkDateTimeAfterSet({ day: 2 }, { hours: 23, minutes: 50 });
    console.log("测试1:", result1); // 结果应为 false，当前日期是16号，时间是11:45，已超过日期，但时间未到达

    // 2. 测试当前日期是否超过8月10号
    var result2 = checkDateTimeAfterSet({ day: 10 }, null);
    console.log("测试2:", result2); // 结果应为 true，当前日期是16号，已超过设定日期

    // 3. 测试当前时间是否超过11:00
    var result3 = checkDateTimeAfterSet(null, { hours: 11, minutes: 0 });
    console.log("测试3:", result3); // 结果应为 true，当前时间是11:45，已超过设定时间

    // 4. 测试当前日期是否超过8月15号，且时间是否在13:30或更晚
    var result4 = checkDateTimeAfterSet({ day: 15 }, { hours: 13, minutes: 30 });
    console.log("测试4:", result4); // 结果应为 false，当前日期是16号，但时间是11:45，尚未到达设定时间

    // 5. 测试当前时间是否超过8:00
    var result5 = checkDateTimeAfterSet(null, { hours: 8, minutes: 0 });
    console.log("测试5:", result5); // 结果应为 true，当前时间是11:45，已超过设定时间

    // 6. 测试当前日期和时间是否超过8月1号23:50
    var result6 = checkDateTimeAfterSet({ day: 1 }, { hours: 23, minutes: 50 });
    console.log("测试6:", result6); // 结果应为 true，当前日期是16号，时间是11:45，已超过设定日期和时间
}






var thread2 = threads.start(function () {
    // while(1){}
    // main()
    // touzi()
    // qiang_tuijian_bao()
    console.log('sdfghjk')

    // 运行测试
    testCheckDateTimeAfterSet();



    thread1.interrupt();
});

// 等待第二个线程完成
thread2.join();

device.vibrate(1000);