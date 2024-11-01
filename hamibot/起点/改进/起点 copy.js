var window1 = null; // 定义在函数外部，以便在函数内部更新

function createFloatyWindow(textContent) {
    // 如果 window1 不为空，关闭之前的浮动窗口
    if (window1 !== null) {
        window1.close();
    }

    // 创建一个新的浮动窗口，并更新 window1
    window1 = floaty.window(
        <frame>
            <text id="text">{textContent}</text>
        </frame>
    );
}

// 调用封装的函数，创建一个浮动窗口
createFloatyWindow("Hello, World!");

for (var i = 0; i < 2; i++) {
    createFloatyWindow(i.toString()); // 将 i 转换为字符串
    sleep(2000);
}
