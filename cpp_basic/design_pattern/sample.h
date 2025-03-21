class Singleton {
public:
    static Singleton& GetInstance();
private:
    Singleton();
    ~Singleton();

    Singleton(const Singleton &single) = delete;

    const Singleton &operator = (const Singleton &single) = delete;

};

Singleton& Singleton::GetInstance() {
    static Singleton instance;
    return instance;
}

Singleton::Singleton() {
    std::cout<<"构造函数"<<std::endl;
}

Singleton::~Singleton() {
    std::cout<<"析构函数"<<std::endl;
}




// c++11之后，静态变量初始化是线程安全的，所以不需要加锁。并且可以通过模版实现通用单例类
template<class T>
class Singleton {
public:
    static T& GetInstance() {
        static T instance;
        return instance;
    }

private:
    Singleton() = default;
    ~Singleton() = default;

    Singleton(const Singleton &single) = delete;

    const Singleton &operator = (const Singleton &single) = delete;
};