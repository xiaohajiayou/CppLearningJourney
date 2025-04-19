
// class Singleton {
// public:
//     static Singleton& GetInstance();
// private:
//     Singleton();
//     ~Singleton();

//     Singleton(const Singleton &single) = delete;

//     const Singleton &operator = (const Singleton &single) = delete;

// };

// Singleton& Singleton::GetInstance() {
//     static Singleton instance;
//     return instance;
// }

// Singleton::Singleton() {
//     std::cout<< "构造函数" <<std::endl;
// }

// Singleton::~Singleton() {
//     std::cout<< "析构函数" <<std::endl;
// }


namespace singleton {

// c++11之后，静态变量初始化是线程安全的，所以不需要加锁。并且可以通过模版实现通用单例类
// 模板类，用于实现单例模式
template<class T>
class Singleton {
public:
    // 获取单例实例的引用
    static T& GetInstance() {
        // 静态局部变量，保证只初始化一次
        static T instance;
        return instance;
    }

private:
    // 构造函数私有化，防止外部实例化
    Singleton();
    // 析构函数私有化，防止外部析构
    ~Singleton();

    // 禁止拷贝构造函数
    Singleton(const Singleton &single) = delete;

    // 禁止赋值操作符
    const Singleton &operator = (const Singleton &single) = delete;
};
}