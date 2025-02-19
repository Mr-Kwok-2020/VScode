var ham = hamibot.env
var ham = {
    "checkbox_01": true,
    "checkbox_02": true,
    "checkbox_03": true,

    "text_03": "第一序列",
    "text_04": "都重生了谁谈恋爱啊丨都市爽文丨搞笑甜宠丨精品多人有声剧"
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




//初始化
function configure() {
    log("开始初始化");
    if (auto.service == null) {
        log("请先开启无障碍服务！");
    } else {
        log("无障碍服务已开启");
        // home()
        sleep(200)

        // console.show();
        log("屏幕尺寸宽、高:", device.width, device.height)
        auto.waitFor();
        console.setTitle("王者营地任务");
        console.setPosition(100, 300)
        console.setSize(800, 2000)

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
function launch_app() {
    log("启动应用------------");
    home()
    // launch("com.qidian.QDReader");
    className("android.widget.TextView").text("王者营地").findOne().click()
    sleep(5000)
    // backHome()
    // clickParentIfClickable(id("imgClose").findOne(1000))
    // waitForActivity('com.qidian.QDReader.ui.activity.MainGroupActivity')
    // backHome()
    log("应用已启动")
}









configure()


var thread1 = threads.start(function () {
    // while (1) {
    //     sleep(1000)
    //     keep_silent()
    // }
});
var thread2 = threads.start(function () {
    main()
    thread1.interrupt();
    while(1){

    }
});

// 测试
var thread3 = threads.start(function () {


});



// 等待第二个线程完成
thread2.join();
// home()
// sleep(5000)
// thread2.interrupt();






function main() {
    
    home()

    launch_app()

    log("所有任务全部完成")
}

