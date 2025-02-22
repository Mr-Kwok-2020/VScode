var ham = hamibot.env
var ham = {
    "checkbox_01": true,
    "checkbox_02": true,
    "checkbox_03": true,

    "text_03": "第一序列",
    "text_04": "都重生了谁谈恋爱啊丨都市爽文丨搞笑甜宠丨精品多人有声剧",


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
    "password": [9, 8, 5, 2, 1, 1]
};


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




function isHomeScreen() {
    // 检测当前界面是否为主页面
    // 是否符合主页面的特征
    sleep(2000)
    home()
    home()
    home()
    if (text("QQ").exists() & text("起点读书").exists() & text("微信").exists()) {
        log("主页面");
        return true
    }
    log("不是主页面");
    sleep(2000)
    return false
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
        sleep(2000);

        // 模拟手势从屏幕底部滑动到屏幕中间
        log("开始上滑进入输入密码");
        gesture(200, [centerX, top], [centerX, bottom]);
        sleep(1000);

        for (var i = 0; i < ham.password.length; i++) {
            var num = ham.password[i];
            var coord = ham.coordinates[num];
            click(coord[0], coord[1]);
            sleep(100); // 短暂等待，确保点击操作有效
        }
        home()
        home()
        sleep(500)

    } while (!isHomeScreen())
    log("解锁成功");
    sleep(2000)
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
    sleep(2000)
    click(850, 2575)
    sleep(2000)
    click(600, 2400)
    sleep(2000)
}
function keep_silent() {
    // log('设置MusicVolume = 0')
    // log(device.getMusicMaxVolume())
    // sleep(5000)
    device.setMusicVolume(0)
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







//初始化
function configure() {
    log("开始初始化");
    if (auto.service == null) {
        log("请先开启无障碍服务！");
    } else {
        log("无障碍服务已开启");
        home()
        sleep(200)
        console.show();
        log("屏幕尺寸宽、高:", device.width, device.height)
        auto.waitFor();
        console.setTitle("王者营地任务");
        console.setPosition(100, 900)//左上角
        console.setSize(800, 1500)//右下角

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
function launch_app_WangZheYingDi() {
    log("启动应用------------");
    home()
    className("android.widget.TextView").text("王者营地").findOne().click()
    sleep(5000)

    log("应用已启动")

    while(!(text("关注").exists() & text("聊天").exists() & text("社区").exists())){
        back()
        sleep(100)
    }


    log("去往：我");
    sleep(200)
    className("android.widget.RadioButton").text("我").findOne().click()

    log("去往：访客列表");
    sleep(200)
    className("android.widget.TextView").text("访客").findOne().click()

    log("去往：浩浩我的神");
    sleep(200)
    className("android.view.View").descStartsWith("浩浩").findOne().click()
    

    log("去往：浩浩我的神 的 访客列表");
    sleep(200)
    id("visitor_text").findOne().click()


}
function getInfoByAimID(aim_ID) {
    log("开始获取信息");

    sleep(2000);
    log("刷新当前页面");

    // 手势操作刷新页面
    gesture(200, [centerX, bottom * 0.25], [centerX, bottom * 0.75]);
    sleep(2000);

    // 查找包含目标ID的控件
    var view = descContains(aim_ID).findOne();
    var fullDesc = view.desc();

    // 使用正则表达式提取方括号内的内容
    var match = fullDesc.match(/\[(.*?)\]/);
    var content = match ? match[1] : "没有找到匹配的内容";

    // 获取当前时间并格式化为 年-月-日 时:分:秒
    var currentDate = new Date();
    var currentTime = currentDate.getFullYear() + "-" +
                      (currentDate.getMonth() + 1).toString().padStart(2, '0') + "-" +
                      currentDate.getDate().toString().padStart(2, '0') + " " +
                      currentDate.getHours().toString().padStart(2, '0') + ":" +
                      currentDate.getMinutes().toString().padStart(2, '0') + ":" +
                      currentDate.getSeconds().toString().padStart(2, '0');

    // 输出总结信息
    // console.log(currentTime + " " + aim_ID + " " + content);

    // 返回对象，包含时间、目标ID和内容
    return {
        time: currentTime,
        aimID: aim_ID,
        content: content
    };
}
function ensureAndAppendHistoryToFile(statusHistory) {
    const filePath = "/sdcard/statusHistory.txt";  // .txt 文件路径

    // 如果文件不存在，创建文件及其父文件夹
    if (!files.exists(filePath)) {
        files.createWithDirs(filePath);  // 创建文件夹和文件
        console.log('文件不存在，已创建新文件！');
    }

    // 将数据转化为文本格式并追加到文件
    statusHistory.forEach(entry => {
        const data = `${entry.time} - ${entry.aimID} - ${entry.content}\n`;  // 按行记录
        files.append(filePath, data);  // 追加到文件
    });

    console.log('状态历史已成功追加到文件！');
}
function getLastStatusFromFile(filePath) {
    // 读取文件内容
    if (files.exists(filePath)) {
        let fileContent = files.read(filePath);
        let lines = fileContent.split("\n");
        if (lines.length > 1) {
            // 获取最后一行内容
            let lastLine = lines[lines.length - 2];  // 最后一行是空行，所以取倒数第二行
            let lastStatus = lastLine.split(" - ")[2];  // 提取最后的状态（根据分隔符“ - ”）
            return lastStatus;
        }
    }
    return null;  // 如果文件为空或不存在返回 null
}








let statusHistory = []; // 用来记录状态变化的时间和状态
configure()
var thread1 = threads.start(function () {
    setScreenBrightness();
    while (1) {
        sleep(1000)
        keep_silent()
    }
});
var thread2 = threads.start(function () {
    main()

    thread1.interrupt();
});
// 测试
var thread3 = threads.start(function () {


});



// 等待第二个线程完成
thread2.join();
// home()
// sleep(5000)
// thread2.interrupt();








function main(){
    unlockScreen()
    clear_back_tasks() 
    launch_app_WangZheYingDi()
    // 主循环
    while (true) {
        var result = getInfoByAimID("养乐多喇叭");

        // 获取文件中最后一条数据的状态
        let lastStatus = getLastStatusFromFile("/sdcard/statusHistory.txt");

        // 如果内容是“游戏在线”或“营地在线”，且与文件中最后一条数据的状态不同，则记录状态
        // if ((result.content === "游戏在线" || result.content === "营地在线" || result.content === "游戏中") &&
        // result.content !== lastStatus) {
        if (result.content !== lastStatus) {
            console.log(result.time, result.aimID, result.content);

            // 将状态历史记录到文件
            let statusHistory = [
                { time: result.time, aimID: result.aimID, content: result.content }
            ];
            ensureAndAppendHistoryToFile(statusHistory);
        }

        sleep(1000);  // 每秒获取一次信息
    }


}











