var ham = hamibot.env
var ham = {
    "checkbox_01": true,
    "checkbox_02": true,
    "checkbox_03": true,
    "checkbox_04": true,
    "text_01": "青山",
    "text_02": "第一侯",
    // "text_03": "从前有个妖怪村"

    "text_03": "第一序列",
    "text_04": "都重生了谁谈恋爱啊丨都市爽文丨搞笑甜宠丨精品多人有声剧"
};

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
//投推荐票
function pull_nan(params) {
    log("男生推荐票投票开始------")
    back_shujia()
    clickParentIfClickable(id("imgClose").findOne(1000))
    clickParentIfClickable(text("任务分组").findOne())
    if (ham.text_01 === '') {
        console.log('没填书');
        return;
    }

    var bookText = textContains(ham.text_01).findOne(500);
    if (bookText === null) {
        console.log('没找到该书');
        return;
    }
    if (!longClickParentIfClickable(bookText)) {
        console.log('投票出现问题请重试');
        return;
    }
    /*if (ca) {
        console.log('投票出现问题请重试');
        ca = false;
        return;
    }*/
    clickParentIfClickable(text('投推荐票').findOne());
    var recommendTicket = textMatches(/拥有\d+主站推荐票/).findOne();
    let votes = jstime(recommendTicket);
    if (votes > 0) {
        clickParentIfClickable(text('全部').findOne());
        clickParentIfClickable(textMatches(/投\d+票/).findOne());
        console.log('已投' + votes + '票');
    } else {
        console.log('没有推荐票');
    }
    backHome();
    log("男生推荐票投票结束")

}
function pull_nv(params) {
    log("女生推荐票投票开始------")
    back_shujia()
    clickParentIfClickable(id("imgClose").findOne(1000))
    clickParentIfClickable(text("任务分组").findOne())
    if (ham.text_02 === '') {
        console.log('没填书');
        return;
    }

    var bookText = textContains(ham.text_02).findOne(500);
    if (bookText === null) {
        console.log('没找到该书');
        return;
    }
    if (!longClickParentIfClickable(bookText)) {
        console.log('投票出现问题请重试');
        return;
    }
    /*if (ca) {
        console.log('投票出现问题请重试');
        ca = false;
        return;
    }*/
    clickParentIfClickable(text('投推荐票').findOne());
    var recommendTicket = textMatches(/拥有\d+女生推荐票/).findOne();
    let votes = jstime(recommendTicket);
    if (votes > 0) {
        clickParentIfClickable(text('全部').findOne());
        clickParentIfClickable(textMatches(/投\d+票/).findOne());
        console.log('已投' + votes + '票');
    } else {
        console.log('没有推荐票');
    }
    backHome();
    log("女生推荐票投票结束")
}
function check_yuepiao(params) {
    log("检测月票数量开始------")
    back_shujia()
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("我的账户").findOne())


    var part = textStartsWith("月票").findOne().parent()
    // 获取该控件的所有子控件
    var children = part.children();


    // log(children.size(),'dsfg')
    // // 遍历并处理每个子控件
    // for (var i = 0; i < children.size(); i++) {
    //     var child = children.get(i);
    //     // 对每个子控件进行处理，比如输出子控件的文本内容
    //     log(child.text(),child.id());
    //     }


    var child = children.get(0)
    // clickParentIfClickable(child)
    var votes = child.text()
    log("检测到月票：", votes)
    sleep(200)

    backHome();
    return votes

}
//签到
function qiandao() {

    console.log("进入签到页面")
    clickParentIfClickable(textStartsWith('签到').findOne(2000))
    clickParentIfClickable(textStartsWith('领福利').findOne(2000))


    var today = new Date();
    var dayOfWeek = today.getDay();
    var thread = threads.start(function () {
        events.observeToast();
        events.onToast(function (toast) {
            let news = toast.getText();
            if (news.indexOf('风险等级') != -1) {
                console.log(news);
                engines.stopAllAndToast()
            }
        });
    });
    //停止线程执行
    thread.interrupt();

    sleep(5000);
    aim_text = "已连续签到"
    console.log('点击',aim_text)
    var part = textStartsWith(aim_text).findOne(2000);
    if (part !== null && part !== undefined) {
        var part = part.parent();
        // 获取该控件的所有子控件
        var children = part.children();
        // handleChildControls(children); // 显示所有子件

        // 获取第一个子控件并点击
        var child = children.get(0);
        clickParentIfClickable(child);
        sleep(2000);
    } else {
        log("未找到符合条件的控件: "+aim_text);
    }



    text("连签说明").waitFor()
    do {
        clickParentIfClickable(text("未领取").findOnce())
    } while (text("未领取").exists());
    back()
    log("wait_1")
    waitForActivity('com.qidian.QDReader.ui.activity.QDBrowserActivity')
    log("wait_2")
    log("wait_3")
    sleep(2000)





    //兑换章节卡
    dayOfWeek = 0
    if (dayOfWeek === 0) {
        log("今天是周日");

        aim_text = "今日已签到"
        console.log('点击',aim_text)
        var part = textStartsWith(aim_text).findOne(2000);
        if (part !== null && part !== undefined) {
            var part = part.parent();
            // 获取该控件的所有子控件
            var children = part.children();
            // handleChildControls(children); // 显示所有子件

            // 获取第一个子控件并点击
            var child = children.get(0);
            clickParentIfClickable(child);
            sleep(2000);
        } else {
            log("未找到符合条件的控件: "+aim_text);
        }



        aim_text = "积攒碎片兑章节卡"
        console.log('点击',aim_text)
        var part = textStartsWith(aim_text).findOne(2000);
        if (part !== null && part !== undefined) {
            var part = part.parent();
            // 获取该控件的所有子控件
            var children = part.children();
            // handleChildControls(children); // 显示所有子件

            // 获取第一个子控件并点击
            var child = children.get(0);
            clickParentIfClickable(child);
            sleep(2000);
        } else {
            log("未找到符合条件的控件: "+aim_text);
        }

 
        console.hide();
        click(1000,2375);
        sleep(2000);
        click(800,1425);
        sleep(3000)
        console.show();

        // aim_text = "30张碎片兑换"
        // console.log('点击',aim_text)
        // var part = textStartsWith(aim_text).findOne(2000);
        // if (part !== null && part !== undefined) {
        //     var part = part.parent();
        //     // 获取该控件的所有子控件
        //     var children = part.children();
        //     // handleChildControls(children); // 显示所有子
        //     // 获取第一个子控件并点击
        //     var child = children.get(0);
        //     clickParentIfClickable(child);
            
        //     click(1000,2375);
        //     sleep(2000);
        //     click(800,1425);
        //     sleep(2000)
            
        // } else {
        //     console.log("未找到符合条件的控件: "+aim_text);
        // }
    } else {
        console.log("今天不是周日");
    }
    console.log("结束签到");
    back()

}
//激励碎片
function looksp() {
    console.log('领碎片开始------------')

    back_shujia()
    clickParentIfClickable(id("imgClose").findOne(1000))
    clickParentIfClickable(text("任务分组").findOne())

    var bookText = textContains(ham.text_03).findOne(500);
    if (bookText === null) {
        console.log('没找到该书');
        return;
    }
    if (!clickParentIfClickable(bookText)) {
        console.log('阅读出现问题请重试');
        return;
    }

    sleep(1000)
    //进入书，开始自动翻页
    //找红包
    while (true) {
        console.log('找红包位置')
        while (true) {
            do {
                click(right - 1, centerY / 15);
            } while (id("tag").exists())
            // console.log('点击屏幕-翻页')
            sleep(500)
            if (text("1个红包").exists()) {
                break
            }
            if (text("0个红包").exists()) {
                break
            }
            // 检测阅读进度重置
            check_if_reset()
        }
        console.log('红包位置已找到')


        if (text("0个红包").exists()) {
            console. log('0个红包')
            break
        }

        console.log('进入点击红包页面')
        clickParentIfClickable(text("1个红包").findOne())
        sleep(2000)

        //再次验证是否有红包可以领取
        if (text("当前章节暂无红包").exists()) {
            console.log('当前章节暂无红包')
            break
        }

        console.log('打开红包')
        clickParentIfClickable(text("马上抢").findOne())

        //看视频
        waitad()
        console.log('领碎片')
        clickParentIfClickable(text("立即领取").findOne(3000))
        back()
        sleep(500)
        click(right - 1, centerY / 15);
        sleep(500)

    }
    console.log('碎片已领完')
    back()
    clickParentIfClickable(text("取消").findOne(500))
    backHome()
}
//福利中心模块
function lookvd() {

    console.log('看视频开始------------')
    clickParentIfClickable(text("我").findOne())

    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("福利中心").findOne())
    console.log("等待福利中心加载")

    text("激励视频任务").waitFor()


    var thread1 = threads.start(function () {
        let stop = textContains("领奖上限").findOne()
        console.log('thread1')
        console.log(stop.text());
        // engines.stopAllAndToast();

        backHome()
        log('进入下一任务')
        return null
    });
    var thread2 = threads.start(function () {
        let stop = textContains("风险等级").findOne()
        console.log('thread2')
        console.log(stop.text());
        // engines.stopAllAndToast();
        backHome()
        console.log('进入下一任务')
        return null
    });

    console.log('下滑页面')
    gesture(2000, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)



    aim_text = "看视频";
    console.log("开始处理", aim_text);
    
    // 最大点击次数
    var maxClicksPerButton = 10;
    
    // 查找所有 "看视频" 按钮
    var buttons = text(aim_text).find();
    if (buttons.empty()) {
        console.log("没有找到任何按钮:", aim_text);
    } else {
        console.log("找到按钮个数:", buttons.size());
    
        // 遍历每个按钮
        for (var i = 0; i < buttons.size(); i++) {
            var button = buttons.get(i);
            console.log(`开始处理第 ${i + 1} 个按钮`);
    
            var clicks = 0;
            while (clicks < maxClicksPerButton) {
                // 点击按钮
                if (clickParentIfClickable(button)) {
                    console.log(`第 ${i + 1} 个按钮已点击 ${clicks + 1} 次`);
                    waitad(); // 等待广告处理完成
                    sleep(1000);
    
                    // 处理弹窗等干扰项
                    var dialog = text("知道了").findOne(500);
                    if (dialog != null) {
                        clickParentIfClickable(dialog);
                        console.log("关闭弹窗");
                    }
                    sleep(1000);
    
                    // 刷新按钮状态
                    buttons = text(aim_text).find();
                    if (i < buttons.size() && buttons.get(i).text() !== aim_text) {
                        console.log(`第 ${i + 1} 个按钮状态已改变`);
                        break; // 跳出当前按钮的处理循环
                    }
                } else {
                    console.log(`无法点击第 ${i + 1} 个按钮`);
                }
    
                clicks++;
                if (clicks >= maxClicksPerButton) {
                    console.log(`第 ${i + 1} 个按钮状态未改变，达到最大点击次数，跳过`);
                }
            }
        }
    }
    console.log("所有按钮处理完成", aim_text);
    
    




    // aim_text = "看视频"
    // console.log('开始',aim_text)
    // var part = textStartsWith(aim_text).findOne(2000);
    // handleChildControls(part.parent().children());
    // bottom_ID = 2
    // console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // while (part.parent().children().get(bottom_ID).text() == '看视频'){
    //     clickParentIfClickable(part.parent().children().get(bottom_ID));
    //     waitad()
    //     sleep(1000);
    //     clickParentIfClickable(text("我知道了").findOne(500))
    //     sleep(1000);
    //     part = textStartsWith(aim_text).findOne(2000);
    //     console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // }    
    // console.log('结束',aim_text)





    // aim_text = "看视频"
    // console.log('开始',aim_text)
    // var part = textStartsWith(aim_text).findOne(2000);
    // handleChildControls(part.parent().parent().children(2));
    // bottom_ID = 2
    // console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // while (part.parent().children().get(bottom_ID).text() == '看视频'){
    //     clickParentIfClickable(part.parent().children().get(bottom_ID));
    //     waitad()
    //     sleep(1000);
    //     clickParentIfClickable(text("我知道了").findOne(500))
    //     sleep(1000);
    //     part = textStartsWith(aim_text).findOne(2000);
    //     console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // }    
    // console.log('结束',aim_text)






    
    // aim_text = "激励视频任务"
    // console.log('开始',aim_text)
    // var part = textStartsWith(aim_text).findOne(2000);
    // handleChildControls(part.parent().children());
    // bottom_ID = 3
    // console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // while (part.parent().children().get(bottom_ID).text() == '看视频'){
    //     clickParentIfClickable(part.parent().children().get(bottom_ID));
    //     waitad()
    //     sleep(1000);
    //     clickParentIfClickable(text("我知道了").findOne(500))
    //     sleep(1000);
    //     part = textStartsWith(aim_text).findOne(2000);
    //     console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // }    
    // console.log('结束',aim_text)


    // aim_text = "看视频领惊喜福利"
    // console.log('开始',aim_text)
    // var part = textStartsWith(aim_text).findOne(2000);
    // handleChildControls(part.parent().children());
    // bottom_ID = 2
    // console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // while (part.parent().children().get(bottom_ID).text() == '看视频'){
    //     clickParentIfClickable(part.parent().children().get(bottom_ID));
    //     waitad()
    //     sleep(1000);
    //     clickParentIfClickable(text("我知道了").findOne(500))
    //     sleep(1000);
    //     part = textStartsWith(aim_text).findOne(2000);
    //     console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // }
    // console.log('结束',aim_text)



    // aim_text = "额外看3次小视频得奖励"
    // console.log('开始',aim_text)
    // var part = textStartsWith(aim_text).findOne(2000);
    // // handleChildControls(part.parent().children());
    // bottom_ID = 2
    // console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // while (part.parent().children().get(bottom_ID).text() == '看视频'){
    //     clickParentIfClickable(part.parent().children().get(bottom_ID));
    //     waitad()
    //     sleep(1000);
    //     clickParentIfClickable(text("我知道了").findOne(500))
    //     sleep(1000);
    //     part = textStartsWith(aim_text).findOne(2000);
    //     console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // }
    // console.log('结束',aim_text)


    // aim_text = "额外看1次小视频得奖励"
    // console.log('开始',aim_text)
    // var part = textStartsWith(aim_text).findOne(2000);
    // // handleChildControls(part.parent().children());
    // bottom_ID = 2
    // console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // while (part.parent().children().get(bottom_ID).text() == '看视频'){
    //     clickParentIfClickable(part.parent().children().get(bottom_ID));
    //     waitad()
    //     sleep(1000);
    //     clickParentIfClickable(text("我知道了").findOne(500))
    //     sleep(1000);
    //     part = textStartsWith(aim_text).findOne(2000);
    //     console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // }
    // console.log('结束',aim_text)


    // aim_text = "看视频获取白泽宇航员装扮"
    // console.log('开始',aim_text)
    // var part = textStartsWith(aim_text).findOne(2000);
    // handleChildControls(part.parent().children());
    // bottom_ID = 2
    // console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // while (part.parent().children().get(bottom_ID).text() == '看视频'){
    //     clickParentIfClickable(part.parent().children().get(bottom_ID));
    //     waitad()
    //     sleep(1000);
    //     clickParentIfClickable(text("我知道了").findOne(500))
    //     sleep(1000);
    //     part = textStartsWith(aim_text).findOne(2000);
    //     console.log('按键状态',part.parent().children().get(bottom_ID).text())
    // }
    // console.log('结束',aim_text)



    //停止线程执行
    thread1.interrupt();
    thread2.interrupt();
    backHome()
}
//卡牌
function kapai() {

    log('卡牌------------')
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("卡牌广场").findOne(5000))
    clickParentIfClickable(text("召唤卡牌").findOne(5000))
    sleep(2000)
    if (text("1次免费").exists()) {
        log('有免费机会开始抽取')
        clickParentIfClickable(text("召唤一次").findOne(5000))
        sleep(3000)
        clickParentIfClickable(text("确定").findOne(5000))
    }
    else {
        log('没到时间')
    }
    backHome()

}
var douyin_cnt = 0
var wait_time = 0
var if_douyin = false
function check_if_douyin() {
    log('检测是否有抖音任务----------')
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("福利中心").findOne())
    
    text("限时彩蛋").waitFor()
    log('已进入福利中心')
    sleep(3000)
    gesture(200, [centerX, bottom * 0.8], [centerX, bottom * 0.2]);
    if (text("分享指定视频到抖音").exists()) {
        if_douyin = true
    }
    if (if_douyin == true) {
        log('存在抖音任务')
    }
    else {
        log('不存在抖音任务')
    }
    backHome()
    return if_douyin
}
function douyin() {
    log('抖音总任务------------')
    do {
        wait_time, douyin_cnt = douyin_danci()
        log('已完成', douyin_cnt, '次')
        if (douyin_cnt == 4) {
            backHome()
            break
        }
        log('本次间隔时间:', wait_time / 1000 / 60, 'min,已完成:', douyin_cnt, '次')


        backHome()
        // wait_time是毫秒，readbook是秒
        readbook(wait_time / 1000)

        launch_qidian_app()

    } while (douyin_cnt < 4)
}
function douyin_danci() {
    log('抖音单次------------')

    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("福利中心").findOne())
    
    text("限时彩蛋").waitFor()
    log('已进入福利中心')
    sleep(3000)
    gesture(200, [centerX, bottom * 0.8], [centerX, bottom * 0.2]);


    // 定位区域
    var douyin_part = textStartsWith('分享指定视频到抖音').findOne().parent()


    // 去分享
    var data = douyin_part.find(text('去分享'))
    if (data.length > 0) {
        log('可以去分享')
        clickParentIfClickable(text("去分享").findOne())
        clickParentIfClickable(text("分享").findOne())
        clickParentIfClickable(text("发布").findOne(1000 * 30))
        sleep(2000)
        clickParentIfClickable(text("发布").findOne(1000 * 30))
        clickParentIfClickable(text("返回起点读书").findOne(1000 * 30))
        backHome();
        log('分享完成')
        return wait_time = 1000 * 60 * 1, douyin_cnt
    }

    // 确认中
    var data = douyin_part.find(text('确认中'))
    if (data.length > 0) {
        backHome();
        log('确认中')
        return wait_time = 1000 * 60 * 1, douyin_cnt
    }

    // 领奖励
    var data = douyin_part.find(text('领奖励'))
    if (data.length > 0) {
        for (let i = data.length - 1; i >= 0; i--) {
            clickParentIfClickable(data[i])
            clickParentIfClickable(text("确认").findOne(2000))  // 检查是否需要
            sleep(500)
        }
        douyin_cnt = douyin_cnt + 1
        var data2 = douyin_part.find(text('已领取'))
        if (data2.length > 0) {
            log('全部完成')
            douyin_cnt = 4
        }
        backHome()
        log('领奖励完成')
        log('插一次卡牌')
        kapai()
        return wait_time = 1000 * 60 * 5, douyin_cnt
    }

    // 提醒我
    var data = douyin_part.find(text('提醒我'))
    if (data.length > 0) {
        backHome();
        log('还未到时间')
        return wait_time = 1000 * 60 * 30, douyin_cnt
    }

    // 已领取
    var data = douyin_part.find(text('已领取'))
    if (data.length > 0) {
        backHome();
        log('确认中')
        return wait_time = 1000 * 60 * 5, douyin_cnt = 4
    }

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
        sleep(2000);

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
    sleep(2000)
    home()
    home()
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
    home()
    // gesture(200, [centerX, bottom * 0], [centerX, bottom * 0.5]);
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
function check_if_reset() {
    // 检测阅读进度重置
    function performClicksIfTextExists(textToCheck, timeout) {
        if (text(textToCheck).exists()) {
            click(centerX, centerY / 30);
            sleep(2000);
            clickParentIfClickable(text("目录").findOne(timeout));
            clickParentIfClickable(text("足迹").findOne(timeout));
            clickParentIfClickable(text("来啦").findOne(timeout));
        }
    }

    performClicksIfTextExists("一个丫鬟，气谁呢", 5000);
    performClicksIfTextExists("加更打赏给月票。", 5000);

    performClicksIfTextExists("是插图的草稿", 5000);
    performClicksIfTextExists("还没画完的插图", 5000);
    performClicksIfTextExists("是插图草稿！", 5000);
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

function listen_book2() {
    log("听书任务开始------")
    back_tingshu()
    clickParentIfClickable(id("imgClose").findOne(1000))
    sleep(500)




    log("点击进入当前听书")
    back_shujia()
    clickParentIfClickable(id("imgClose").findOne(1000))
    clickParentIfClickable(text("任务分组").findOne())

    // var bookText = textContains(ham.text_04).findOne(500);
    // if (bookText === null) {
    //     console.log('没找到该书');
    //     return;
    // }
    // clickParentIfClickable(bookText)
    sleep(5000)
    click(500, 450);
    sleep(5000)




    clickParentIfClickable(text("目录").findOne())
    sleep(5000)
    clickParentIfClickable(text("去底部").findOne())
    sleep(5000)
    clickParentIfClickable(text("去顶部").findOne())
    sleep(5000)
    clickParentIfClickable(text("【发刊词】原著作者错哪儿了推荐语").findOne())
    // click(500, 450);
    sleep(5000)
    backHome()

    // 领奖励

    back_tingshu()
    clickParentIfClickable(id("imgClose").findOne(1000))
    sleep(2000)
    // 进入天天福利

    clickParentIfClickable(text("天天福利").findOne())
    sleep(5000)


    // 正式会员
    var currentTime = get_current_time();
    if (currentTime.hours >= 3 && currentTime.minutes >= 0) {
        click(1000, 1650);
        sleep(5000)
    }

    // 签到打卡
    click(1000, 1850);
    sleep(5000)
    // 查看小程序专属福利啦
    click(1000, 2120);
    sleep(5000)
    home();// 回主页面
    sleep(5000)
    click(1025, 1000);// 回起点
    sleep(5000)
    click(1000, 2120);// 领取
    sleep(5000)

    // 隐藏控制台
    console.hide();
    sleep(2000)
    click(160, 1150);
    sleep(2000)
    click(600, 1620);
    sleep(1000)

    click(380, 1150);
    sleep(2000)
    click(600, 1620);
    sleep(1000)

    click(600, 1150);
    sleep(2000)
    click(600, 1620);
    sleep(1000)

    click(1050, 1150);
    sleep(2000)
    click(600, 1620);
    sleep(1000)

    // 隐藏控制台
    console.show();
    sleep(2000)
    backHome()



}

// 检查时间是否超过当日的23:50分
function checkTime() {
    var currentTime = get_current_time();
    if (currentTime.hours >= 23 && currentTime.minutes >= 40) {
        return true;
    }
    return false;
}

// 检查时间是否超过当日的23:50分
function checkTime2() {
    var currentTime = get_current_time();
    if (currentTime.hours >= 8 && currentTime.minutes >= 0) {
        return true;
    }
    return false;
}
function checkDateAfterSet() {
    // 获取当前日期
    var now = new Date();
    // 获取当前日期的天数
    var day = now.getDate();
    log('当前日期为：', day)

    // 检查当前日期是否大于2号
    if (day > 1) {
        return true;
    } else {
        return false;
    }
}
function mo_yue_redbag() {
    log("摸红包------")

    clickParentIfClickable(text("发现").findOne())
    sleep(500)
    clickParentIfClickable(id("imgClose").findOne(1000))
    sleep(500)
    clickParentIfClickable(text("红包广场").findOne(1000))
    sleep(500)
    clickParentIfClickable(text("月票").findOne(1000))
    sleep(500)

    // 隐藏控制台
    console.hide();
    do {
        gesture(200, [centerX, bottom * 0.4], [centerX, bottom * 0.9]);
        sleep(500)

    } while (!text("马上抢").exists())

    clickParentIfClickable(text("马上抢").findOne(1000))
    clickParentIfClickable(id("btnHongbaoGet").findOne(1000))

    // 使手机震动1秒（1000毫秒）
    device.vibrate(1000);
    sleep(1000 * 10)
    backHome()
    log("抢到红包------")
    // 隐藏控制台
    console.show();
}
function mo_tuijian_redbag() {

    log("摸红包------")
    keep_silent()
    clickParentIfClickable(text("发现").findOne())
    sleep(500)
    clickParentIfClickable(id("imgClose").findOne(1000))
    sleep(500)
    clickParentIfClickable(text("红包广场").findOne(1000))
    sleep(500)
    clickParentIfClickable(text("推荐票").findOne(1000))
    sleep(500)

    // 隐藏控制台
    console.hide();
    do {
        gesture(200, [centerX, bottom * 0.4], [centerX, bottom * 0.9]);
        sleep(500)

    } while (!text("马上抢").exists())

    clickParentIfClickable(text("马上抢").findOne(1000))
    clickParentIfClickable(id("btnHongbaoGet").findOne(1000))

    // 使手机震动1秒（1000毫秒）
    device.vibrate(1000);
    sleep(1000 * 10)
    backHome()
    log("抢到红包------")
    // 显示控制台
    console.show();
}


configure()


var thread1 = threads.start(function () {
    setScreenBrightness();
    while (1) {
        sleep(1000)
        keep_silent()
    }
});
var yuepiaocnt = 0
var thread2 = threads.start(function () {
    main222()
    
    thread1.interrupt();
});

// 测试
var thread3 = threads.start(function () {


});

// // 等待第二个线程完成
// thread2.join();

// home()
// home()
// sleep(5000)
// lockScreen()

// thread2.interrupt();






function main() {
    // unlockScreen()
    // launch_qidian_app()

    // qiandao()
    // looksp()
    // lookvd()
    // playgame()

    // getPrize()
    // buy()
    // kapai()

    // listenToBook()

    // if_douyin = check_if_douyin()
    // // if(if_douyin == true){
    // //     douyin()
    // // }
    var tvVipTime = 0
    tvVipTime = get_read_jifen()
    while (tvVipTime < 3100) {
        log('当天阅读时间：', tvVipTime)
        readbook(30 * 60)
        kapai()
        // listen_book2()
        if (checkTime2() & yuepiaocnt > 0 & checkDateAfterSet()) {
            log("符合抢红包日期和时间");
            log("还剩月票：", yuepiaocnt);

            mo_yue_redbag()

            yuepiaocnt = check_yuepiao()
        }
        tvVipTime = get_read_jifen()
    }
    log("所有任务全部完成")
}




function main222() {
    // unlockScreen()
    // launch_qidian_app()

    // qiandao()
    // looksp()
    lookvd()
    playgame()
    kapai()

    // while (true) {
    //     readbook(30 * 60)
    //     kapai()

      
    //     get_read_jifen()
    // }
    log("所有任务全部完成")
}
function touzi() {
    var read_book_time = 13 * 60
    var strat_read_book_time = get_timestamp()
    var if_print_cnt = 0
    log("阅读开始时间：", get_current_time())
    log("持续时间：", read_book_time / 60, '分钟')

    sleep(1000)
    log("投资：")
    //进入书，开始自动翻页
    while (true) {
        click(right - 1, centerY / 20);
        sleep(5000)

        click(0 + 1, centerY / 20);
        sleep(5000)

        if_print_cnt = if_print_cnt + 1

        var item = read_book_time - (get_timestamp() - strat_read_book_time)
        // 倒计时计算
        if (if_print_cnt >= 6) {
            log('还剩余', item / 60, '分钟')
            if_print_cnt = 0
        }
        // 检测什么时候停止阅读
        if (get_timestamp() - strat_read_book_time > read_book_time) {
            break
        }
    }
    log('结束阅读，时长：', read_book_time / 60, '分钟')

    // 使手机震动1秒（1000毫秒）
    device.vibrate(5000);
}
function qiang_tuijian_bao() {
    var bao_cnt = 0
    var max_cnt = 4

    unlockScreen()
    launch_qidian_app()

    while (bao_cnt < 4) {
        log("抢红包");

        mo_tuijian_redbag()
        bao_cnt = bao_cnt + 1
    }
    log("所有任务全部完成")


}




function readbook(read_book_time) {

    log("阅读开始------")

    back_shujia()
    clickParentIfClickable(id("imgClose").findOne(1000))
    clickParentIfClickable(text("任务分组").findOne())

    var strat_read_book_time = get_timestamp()
    log("阅读开始时间：", get_current_time())
    log("持续时间：", read_book_time / 60, '分钟')


    var bookText = textContains(ham.text_03).findOne(500);
    if (bookText === null) {
        console.log('没找到该书');
        return;
    }
    if (!clickParentIfClickable(bookText)) {
        console.log('阅读出现问题请重试');
        return;
    }

    sleep(1000)
    //进入书，开始自动翻页
    while (true) {
        do {
            click(right - 1, centerY / 20);
        } while (id("tag").exists())


        sleep(5000)
        // 检测阅读进度重置
        check_if_reset()


        var item = read_book_time - (get_timestamp() - strat_read_book_time)
        // 倒计时计算
        if (item % (60 * 5) == 0) {
            log('还剩余', item / 60, '分钟')
        }
        // 检测什么时候停止阅读
        if (get_timestamp() - strat_read_book_time > read_book_time) {
            break
        }
    }
    log('结束阅读，时长：', read_book_time / 60, '分钟')
    backHome()
}
//等待广告
function waitad() {

    log('看广告')
    // 广告时间对象
    var reward
    //等待广告出现
    while (className("android.view.View").depth(4).exists()) {
        sleep(500)
    }
    //等待广告时间对象
    reward = textEndsWith("，可获得奖励").findOne(5000)
    if (reward == null) {
        if (className("android.view.View").depth(4).exists()) {
            while (className("android.view.View").depth(4).exists()) {
                sleep(500)
            }
            if (!textEndsWith("，可获得奖励").exists()) {
                back()
                sleep(500)
                console.log('广告未加载');
                return
            }
        } else if (className("android.view.View").depth(5).exists()) {
            back()
            sleep(500)
            console.log('广告未加载');
            return
        } else {
            console.log('未进入广告页面');
            return
        }
    }
    //等待广告出现
    while (className("android.view.View").depth(4).exists()) {
        sleep(500)
    }
    if (!textEndsWith("，可获得奖励").exists()) {
        back()
        sleep(500)
        console.log('广告未加载');
        return
    }
    //获取关闭坐标
    var gb = text("关闭").findOne(400)
    var cross = text("cross").findOne(400)
    var tg = text("跳过广告").findOne(400)
    // var wz = text("此图片未加标签。打开右上角的“更多选项”菜单即可获取图片说明。").findOnce()
    var zb = null
    if (gb) {
        zb = gb
    } else if (cross) {
        zb = cross
    } else if (tg) {
        zb = tg
    } /*else if (wz) {
        zb = wz
    }*/
    if (zb == null) {
        console.log('获取关闭坐标')
        var video_quit = reward.bounds()
        var x1 = 0;
        var x2 = video_quit.left;
        var y1 = video_quit.top;
        var y2 = video_quit.bottom;
        X = parseInt((x1 + x2) / 2)
        Y = parseInt((y1 + y2) / 2)
        // var nocross = true
    }

    // 获取等待时间
    var time = jstime(textEndsWith("，可获得奖励").findOne())
    if (time == null) {
        log('获取不到时间，重新获取')
        log('点击退出')
        do {
            if (!textEndsWith("，可获得奖励").exists()) {
                back()
                sleep(500)
                console.log('获取不到坐标')
                return
            }
            if (zb == null) {
                click(X, Y)
            } else {
                clickParentIfClickable(zb)
            }
            sleep(500)
        } while (!text("继续观看").exists())
        time = jstime(textEndsWith("可获得奖励").findOne())
        clickParentIfClickable(text("继续观看").findOne())
        if (time == null) {
            time = textMatches(/\d+/).findOnce()
            if (time) {
                time = parseInt(time.text())
            }
        }
    }

    //等待广告结束
    var num
    if (time) {
        log('等待' + (time + 1) + '秒')
        sleep(1000 * (time + 1))
        num = 0
        do {
            if (zb == null) {
                click(X, Y)
            } else {
                clickParentIfClickable(zb)
            }
            if (clickParentIfClickable(text("继续观看").findOne(500))) {
                sleep(1000)
                num++
                log('等待' + num + '秒')
            }
        } while (textEndsWith("，可获得奖励").exists());
    } else {
        //获取不到时间
        log('等待视频结束')
        // clickParentIfClickable(text("继续观看").findOne())
        num = 0
        do {
            num++
            sleep(1000)
            log('等待' + num + '秒')
        } while (textEndsWith("，可获得奖励").exists());
    }
    //判断是否还在广告页面
    if (className("android.view.View").depth(5).exists()) {
        back()
        sleep(500)
    }
    log('广告结束')
    sp++
    log('已看视频' + sp + '个')
}
//听书
function listenToBook() {
    log("开始-听书------------")

    clickParentIfClickable(text("我").findOne())

    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("福利中心").findOne())
    
    text("限时彩蛋").waitFor()
    log("等待福利中心加载")
    sleep(2000)

    var bookV
    // let listenTime
    bookV = textContains("当日听书").findOne(1000)
    if (bookV == null) {
        console.log('没有听书')
        log("结束-听书")
        backHome()
        return
    }
    // let listeningTime = jstime(bookV);
    // if (textContains("当日玩游戏").findOnce() == null) {
    //      listenTime = jstime(bookVs);
    // }
    bookV = bookV.parent()
    if (clickParentIfClickable(bookV.findOne(text('去完成'))) != null) {
        log("正在进行听书ing")
        sleep(1000 * 75)
        back()
        clickParentIfClickable(text("取消").findOne())
        // back()
    }
    log("结束-听书")
    backHome()
}
//玩游戏
function playgame() {
    log("开始-玩游戏------------")

    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("福利中心").findOne())
    text("激励视频任务").waitFor()
    log("等待福利中心加载")

    console.log('下滑页面')
    gesture(500, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)
    gesture(500, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)
    gesture(500, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)

    
    aim_text = "当日玩游戏10分钟"
    console.log('开始',aim_text)
    var part = textStartsWith(aim_text).findOne(2000);
    // handleChildControls(part.parent().children());
    bottom_ID = 3
    console.log('按键状态',part.parent().children().get(bottom_ID).text())
    if (part.parent().children().get(bottom_ID).text() != '去完成') {
        console.log('没有游戏可玩')
        backHome()
        console.log("结束-玩游戏")
        return
    }

    game = part.parent()
    let finishing
    var pt = 10
    device.keepScreenDim();
    while ((finishing = game.findOne(text('去完成'))) != null) {

        // var repetitions = 4
        do {

            if (!clickParentIfClickable(finishing)) {
                back()
                clickParentIfClickable(text("游戏中心").findOne())
            }
            sleep(500)
        } while (textContains("当日玩游戏").exists());
        log("前往游戏中心")
        log('test1')

        //防止网络不好，加载不出来
        textContains("游戏中心").waitFor()


        log('test2')
        if (clickParentIfClickable(text("排行").findOne(5000)) == null) {
            clickParentIfClickable(text("在线玩").findOne())
        } else {
            text("新游榜").waitFor()
            text("热门榜").waitFor()
            text("畅销榜").waitFor()
            clickParentIfClickable(text("热门榜").findOne())
            clickParentIfClickable(text("在线玩").findOne())
            // repetitions++
        }
        log("进入游戏")
        log('剩余' + (pt + 0.5) + '分钟')
        startCountdown(pt + 0.5)


    }
    // device.cancelKeepingAwake();
    backHome()


    // 领取奖励
    console.log('游戏领取奖励')
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("福利中心").findOne())
    
    text("激励视频任务").waitFor()
    log("等待福利中心加载")

    console.log('下滑页面')
    gesture(500, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)
    gesture(500, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)
    gesture(500, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)

    aim_text = "当日玩游戏10分钟"
    console.log('开始',aim_text)
    var part = textStartsWith(aim_text).findOne(2000);
    // handleChildControls(part.parent().children());
    bottom_ID = 3
    console.log('按键状态',part.parent().children().get(bottom_ID).text())
    clickParentIfClickable(part.parent().children().get(bottom_ID));
    clickParentIfClickable(text("我知道了").findOne(500))



    
    backHome()
    log("结束-玩游戏")





}
//倒计时
function startCountdown(minutes) {
    var count = minutes * 60; // 倒计时的秒数
    var remainingMinutes
    var remainingSeconds
    for (var i = count; i >= 0; i--) {
        remainingMinutes = Math.floor(i / 60); // 剩余分钟数
        remainingSeconds = i % 60; // 剩余秒数
        //清除控制台
        console.clear()
        // 每分钟提示倒计时
        if (i > 60) {
            log("倒计时还剩 " + remainingMinutes + " 分钟 " + remainingSeconds + " 秒 ");
        }
        // 剩余60秒钟提示倒计时
        if (i <= 60) {
            log("倒计时还剩 " + i + " 秒");
        }
        sleep(1000); // 等待1秒
        device.wakeUpIfNeeded();
    }
    console.clear()
    log("倒计时已结束");
}
//领取
function getPrize() {
    log("开始-领取奖励------------")
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("福利中心").findOne())

    log("等待福利中心加载")
    text("激励视频任务").waitFor()

    console.log('下滑页面')
    gesture(2000, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)


    var prizePool
    prizePool = text("领奖励").find()
    for (i = 0; i < prizePool.length; i++) {
        // prizePool[i].click()
        clickParentIfClickable(prizePool[i])
        clickParentIfClickable(text("我知道了").findOne(750))
    }
    clickParentIfClickable(id("ivClose").findOne(500))
    backHome()
    console.log('完成-领奖励')

}
//兑换
function buy() {
    console.log('开始碎片兑换------------')
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("福利中心").findOne())
    log("等待福利中心加载")
    
    text("限时彩蛋").waitFor()

    log("等待下滑")
    sleep(1000)    
    gesture(200, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)


    clickParentIfClickable(desc('更多好礼').findOne())
    sleep(3000)







    console.log('检索-畅享卡')
    text('畅享卡').waitFor()
    var enjoyCard = textStartsWith('7天').findOne().parent().parent()
    var convertibleList = enjoyCard.find(text('兑换'))
    log('等待兑换个数：', convertibleList.length)
    if (convertibleList.length > 0) {
        for (let i = convertibleList.length - 1; i >= 0; i--) {
            clickParentIfClickable(convertibleList[i])
            clickParentIfClickable(text("确认").findOne(2000))
            sleep(2000)
        }
        console.log('已兑换-畅享卡')
    }
    else {
        console.log('没有兑换到-畅享卡')
    }


    gesture(200, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)
    gesture(200, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(500)
    gesture(200, [centerX, bottom * 0.75], [centerX, bottom * 0.25]);
    sleep(5000)


    console.log('100-50')
    var enjoyCard = textStartsWith('满100点-50点订阅券').findOne().parent()
    var convertibleList = enjoyCard.find(text('兑换'))
    log('等待兑换个数：', convertibleList.length)
    if (convertibleList.length > 0) {
        for (let i = convertibleList.length - 1; i >= 0; i--) {
            clickParentIfClickable(convertibleList[i])
            clickParentIfClickable(text("确认").findOne(2000))
            sleep(2000)
        }
        console.log('已兑换-100-50')
    }
    else {
        console.log('没有兑换到-100-50')
    }



    console.log('检索-卡牌召唤券')
    var enjoyCard = textStartsWith('卡牌召唤券').findOne().parent()
    var convertibleList = enjoyCard.find(text('兑换'))
    log('等待兑换个数：', convertibleList.length)
    if (convertibleList.length > 0) {
        for (let i = convertibleList.length - 1; i >= 0; i--) {
            clickParentIfClickable(convertibleList[i])
            clickParentIfClickable(text("确认").findOne(2000))
            sleep(2000)
        }
        console.log('已兑换-卡牌召唤券')
    }
    else {
        console.log('没有兑换到-卡牌召唤券')
    }



    sleep(2000)

    backHome()

}
// 阅读积分
function get_read_jifen() {
    log('阅读积分------------')
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("阅读积分").findOne())
    
    // text("每日阅读领积分").waitFor()
    clickParentIfClickable(text("关闭").findOne(1000))
    sleep(3000)
    console.log('到达指定页面')
    

    // 隐藏控制台  领取积分
    console.hide();
    sleep(200)
    //550积分
    click(1000, 730);
    sleep(3000)
    // click(425, 900);
    // sleep(1000)
    // click(650, 900);
    // sleep(1000)
    // click(860, 900);
    // sleep(1000)
    // click(1090, 900);
    // sleep(1000)


    console.hide();
    sleep(200)
    click(1000, 1250);
    sleep(1000)
    backHome()
    console.show();
    clickParentIfClickable(text("我").findOne())
    clickParentIfClickable(text("我知道了").findOne(1000))
    clickParentIfClickable(text("阅读积分").findOne())
    // text("每日阅读领积分").waitFor()
    clickParentIfClickable(text("关闭").findOne(1000))
    sleep(3000)
    console.log('到达指定页面')




    console.hide();
    sleep(200)
    click(1000, 1475);
    sleep(1000)
    backHome()



    //显示控制台
    console.show();



}
function back_shujia() {

    var part = textStartsWith("发现").findOne().parent().parent().parent()
    // 获取该控件的所有子控件
    var children = part.children();


    log('返回书架，子目标数：', children.size())
    // // 遍历并处理每个子控件
    // for (var i = 0; i < children.size(); i++) {
    //     var child = children.get(i);
    //     // 对每个子控件进行处理，比如输出子控件的文本内容
    //     log(child.text(),child.id());
    //     }


    var child = children.get(0)
    clickParentIfClickable(child)
}

function back_tingshu() {

    var part = textStartsWith("发现").findOne().parent().parent().parent()
    // 获取该控件的所有子控件
    var children = part.children();


    log('返回听书，子目标数：', children.size())
    // // 遍历并处理每个子控件
    // for (var i = 0; i < children.size(); i++) {
    //     var child = children.get(i);
    //     // 对每个子控件进行处理，比如输出子控件的文本内容
    //     log(child.text(),child.id());
    //     }


    var child = children.get(2)
    clickParentIfClickable(child)
}



// 遍历并处理子控件的函数
function handleChildControls(children) {
    for (var i = 0; i < children.size(); i++) {
        var child = children.get(i);
        // 对每个子控件进行处理，比如输出子控件的文本内容和 ID
        log(child.text(), child.id());
    }
}
