var ham = hamibot.env
// var ham = {
//     "mode": 0,//
//     "targetTemperature": 22
// };
// var mode_list = ['关闭','制冷模式','制热模式','自动模式','通风模式','除湿模式']

// 定义每个数字的坐标
var coordinates = {
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
};
// 输入解锁密码 '985211'

var password = [9, 8, 5, 2, 1, 1];
var bounds

var centerX
var centerY

var left
var right
var top
var bottom

var sp = 0
var X
var Y
var InitialValue = null

//初始化
function configure() {
    log("开始初始化");
    if (auto.service == null) {
        log("请先开启无障碍服务！");
    } else {
        log("无障碍服务已开启");
        // home()
        sleep(200)


        // 关闭之前打开的信息输出窗口
        // console.hide();
        // click(550,570);
        // console.clear();


        console.show();
        log("屏幕尺寸宽、高:", device.width, device.height)
        auto.waitFor();
        console.setTitle("起点自动任务");
        console.setPosition(100, 300)
        console.setSize(800, 1300)

        bounds = className("android.widget.FrameLayout").depth(0).findOne()
        centerX = getXy(bounds).centerX;
        centerY = getXy(bounds).centerY;
        right = bounds.bounds().right
        left = bounds.bounds().left
        top = bounds.bounds().top
        bottom = bounds.bounds().bottom
        log("元素的左边界: " + left);
        log("元素的上边界: " + top);
        log("元素的右边界: " + right);
        log("元素的下边界: " + bottom);
        log("元素的中心X: " + centerX);
        log("元素的中心Y: " + centerY);

        log("初始化结束");
    }
}
//启动应用
function launch_qidian_app() {
    log("启动应用------------");
    home()
    clear_back_tasks()
    // launch("com.qidian.QDReader");
    className("android.widget.TextView").text("起点读书").findOne().click()
    sleep(5000)
    // backHome()
    clickParentIfClickable(id("imgClose").findOne(1000))
    // waitForActivity('com.qidian.QDReader.ui.activity.MainGroupActivity')
    backHome()
    log("应用已启动")
}



// 工具
function jstime(textObj) {
    if (textObj == null) {
        return null
    }
    // 存储初始文本内容
    var initText = textObj.text();
    // log(initText)
    //获取时间
    var match = initText.match(/\d+/g);
    return match ? parseInt(match[0]) : null;
}
function getXy(obj) {
    if (obj == null) {
        return null;
    }
    var bounds = obj.bounds();
    return {
        centerX: (bounds.left + bounds.right) / 2,
        centerY: (bounds.top + bounds.bottom) / 2
    };
}
function clickCenter(params) {
    var center = getXy(params);
    if (center == null) {
        console.log('没找到')
        return
    }
    click(center.centerX, center.centerY);
    console.log('点击坐标')
}
function backHome(params) {
    do {
        back()
    } while (id("normal").findOne(500) == null)

    back_shujia()
    sleep(1000)
    log('已到主界面');
}
var cnt = 0
function clickParentIfClickable(widget) {

    if (InitialValue == null) {
        InitialValue = widget
    }
    if (widget === null) {
        console.log('找不到');
        InitialValue = null
        return null;  // 终止递归的条件：如果 widget 是空值，则结束递归
    }
    if (widget.click()) {
        console.log('已点击');
        InitialValue = null

        sleep(100); // 等待1秒
        return true;  // 点击控件
    }
    var parentWidget = widget.parent();  // 获取控件的父类
    if (parentWidget === null) {
        console.log('不可点击');
        clickCenter(InitialValue)
        InitialValue = null
        return false;
    }
    return clickParentIfClickable(parentWidget);
    // 递归调用自身，传入父类控件进行下一次查找和点击
}
function longClickParentIfClickable(widget) {
    if (widget === null) {
        console.log('找不到');
        return null;  // 终止递归的条件：如果 widget 是空值，则结束递归
    }
    if (widget.longClick()) {
        console.log('已长按');
        return true;  // 点击控件
    }
    var parentWidget = widget.parent();  // 获取控件的父类
    if (parentWidget === null) {
        console.log('不可长按');
        return false
    }
    return longClickParentIfClickable(parentWidget);  // 递归调用自身，传入父类控件进行下一次查找和点击
}
function unlockScreen() {
    log("开始解锁屏幕--------")
    if (device.isScreenOn()) {
        log('当前屏亮先锁屏')
        lockScreen()
    }
    do {
        // 唤醒设备
        device.wakeUpIfNeeded();
        sleep(500);

        // 模拟手势从屏幕底部滑动到屏幕中间
        log("开始上滑进入输入密码");
        gesture(200, [centerX, top], [centerX, bottom]);
        sleep(300);

        for (var i = 0; i < password.length; i++) {
            var num = password[i];
            var coord = coordinates[num];
            click(coord[0], coord[1]);
            sleep(100); // 短暂等待，确保点击操作有效
        }
        home()
        home()
        sleep(500)

    } while (!isHomeScreen())
    log("解锁成功");
    sleep(1000)
    home()
    home()
}
function lockScreen() {
    // // 通过发送广播来锁定屏幕
    // // 注意：这可能需要设备是 root 或特权应用
    // let intent = new Intent("android.intent.action.SCREEN_OFF");
    // context.sendBroadcast(intent);
    home()
    home()
    var success = runtime.accessibilityBridge.getService().performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_LOCK_SCREEN)

}
function clear_back_tasks() {
    log('清除后台应用')
    home()
    home()
    gesture(200, [centerX, bottom * 0], [centerX, bottom * 0.5]);
    sleep(300)
    click(850, 2575)
    sleep(300)
    click(600, 2400)
    sleep(2000)
}
function keep_silent() {
    // log('设置MusicVolume = 0')
    // log(device.getMusicMaxVolume())
    // sleep(5000)
    device.setMusicVolume(0)
}
function get_timestamp() {
    // 获取当前时间戳
    var timestampInSeconds = Math.floor(new Date().getTime() / 1000);
    // // 显示时间戳
    // log("当前时间戳: " + timestampInSeconds );
    return timestampInSeconds
}
function get_timestamp222() {
    // 获取当前时间戳
    var timestampInSeconds = Math.floor(new Date().getTime());
    // // 显示时间戳
    // log("当前时间戳: " + timestampInSeconds );
    return timestampInSeconds
}
function setScreenBrightness() {
    log('开启自动管理屏幕光照强度')


    // 初始化亮度
    var Brightness_value = device.getBrightness();

    // 设置光线传感器监听器
    var sensor = sensors.register('light');
    sensor.on('change', (event, light) => {

        if (device.getBrightnessMode() == 1) {
            device.setBrightnessMode(0)
        }

        var lightIntensity = Math.floor(light); // 将光强转换为整数
        var min_limit = 6
        var max_limit = 50
        if (lightIntensity < min_limit) {
            Brightness_value = 0;
        } else if (lightIntensity > max_limit) {
            Brightness_value = 225;
        } else {
            // 插值计算
            Brightness_value = Math.floor((lightIntensity - min_limit) / (max_limit - min_limit) * (225 - 0));
        }

        // 设置屏幕亮度
        device.setBrightness(Brightness_value);

        // log('光强：', lightIntensity, '设置亮度：', Brightness_value);
    });
}
function get_current_time() {
    var now = new Date();

    var hours = now.getHours(); // 获取当前小时
    var minutes = now.getMinutes(); // 获取当前分钟
    var seconds = now.getSeconds(); // 获取当前秒
    var milliseconds = now.getMilliseconds(); // 获取当前毫秒

    log("当前时间: " + hours + ":" + minutes + ":" + seconds + "." + milliseconds);

    return {
        hours: hours,
        minutes: minutes,
        seconds: seconds,
        milliseconds: milliseconds
    };
}
function isHomeScreen() {
    // 检测当前界面是否为主页面
    // 是否符合主页面的特征
    if (text("王者荣耀").exists() & text("起点读书").exists() & text("微信").exists()) {
        log("主页面");
        return true
    }
    log("不是主页面");
    sleep(2000)
    return false
}
// 封装成函数：检查当前模式
function checkCurrentMode() {
    // 查找 ID 为 'mode' 的控件，并获取其文本内容
    var modeControl = id("mode").findOne(2000); // 等待2秒查找控件

    if (modeControl != null) {
        var modeText = modeControl.text(); // 获取控件的文本内容
        log("检查当前模式是: " + modeText);
        return modeText; // 返回获取的模式文本
    } else {
        log("未找到ID为'mode'的控件");
        return null; // 返回 null 以表示未找到控件
    }
}
function change_mode(setMode){
    // 开始循环检查模式
    while (true) {
        // 检查当前模式是否与设定模式一致
        if (setMode != checkCurrentMode()) {
            log("更改模式...");
            clickParentIfClickable(id("btn_air_condition_mode").findOne(1000));
            sleep(500); // 等待半秒以确保按钮点击反应
        } else {
            log("当前模式与设定模式一致，无需更改。");
            change_temp()
            break;
        }
        sleep(500); // 每次循环间等待2秒
    }
}
function change_temp(){

    // 设定目标温度
    var targetTemperature = ham.targetTemperature; // 设定目标温度
    while (true) {
        // 获取当前温度控件
        var temperature_ = id("temperature").findOne(2000); // 等待2秒查找控件
        if (temperature_ != null) {
            var temperature_var = temperature_.text(); // 获取控件的文本内容
            log("当前温度是: " + temperature_var);

            // 将字符串转换为数字
            var currentTemperature = parseFloat(temperature_var);

            // 判断是否已达到目标温度
            if (currentTemperature == targetTemperature) {
                log("温度已达到设定值: " + targetTemperature + "，操作停止。");
                break; // 退出循环
            }

            // 如果温度高于目标温度，点击降低温度按钮
            if (currentTemperature > targetTemperature) {
                // log("温度高于目标值，正在降低温度...");
                clickParentIfClickable(id("btn_temperature_down").findOne(1000));
            }

            // 如果温度低于目标温度，点击升高温度按钮
            else if (currentTemperature < targetTemperature) {
                // log("温度低于目标值，正在升高温度...");
                clickParentIfClickable(id("btn_temperature_up").findOne(1000));
            }

            // 等待一段时间后再次检测温度
            sleep(500); // 每次循环间等待2秒
        } else {
            log("未找到温度控件，重新尝试...");
        }

        // 每次调节后等待一段时间再重新获取温度
        sleep(500); 
    }
}

configure()
unlockScreen()
log("启动应用------------");
home()
clear_back_tasks()
className("android.widget.TextView").text("智能遥控").findOne().click()
sleep(2000)
clickParentIfClickable(id("item_text").findOne(1000))
log('进入遥控器')

// device.vibrate(1000);
sleep(500)
// 主逻辑
var modes = checkCurrentMode(); // 调用函数获取当前模式和设定模式

if (modes) {
    var currentMode = modes;
    var setMode = ham.mode;
    var targetTemperature = ham.targetTemperature; // 设定目标温度
    log('dfghj',currentMode,targetTemperature,)

    // 处理逻辑
    if (currentMode !== "关闭" && setMode === "关闭") {
        clickParentIfClickable(id("btn_power_toggle").findOne(1000));
        sleep(500)
        log("关闭空调开关");
    } else if (currentMode !== "关闭" && setMode !== "关闭") {
        change_mode(setMode); // 当前模式为非关闭且设定模式也非关闭，进行模式更改
    } else if (currentMode === "关闭" && setMode === "关闭") {
        log("空调已经处于关闭状态，直接退出后续处理。");
    } else if (currentMode === "关闭" && setMode !== "关闭") {
        clickParentIfClickable(id("btn_power_toggle").findOne(1000));
        log("开启空调开关");
        change_mode(setMode); // 开启空调后更改模式
    }
}


home()
sleep(5000)
lockScreen()
console.hide()
