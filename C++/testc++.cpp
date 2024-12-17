#include <iostream> // 用于输入输出
#include <chrono>   // 用于时间
#include <thread>   // 用于线程



int main() {
    // 延迟 10 秒
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 输出 "你好，世界" 到控制台
    std::cout << "你好，世界" << std::endl;
    // 延迟 10 秒
    std::this_thread::sleep_for(std::chrono::seconds(2));
    // 输出 "你好，世界" 到控制台
    std::cout << "你好，世界" << std::endl;

    // 返回 0 表示程序正常结束
    return 0;
}
/* 用于输出 Hello World 的注释
 
cout << "Hello World"; // 输出 Hello World
 
*/

#if condition
  code1
#else
  code2
#endif