## 1. 什么是单例模式
单例模式是指在整个系统生命周期内，保证一个类只能产生一个实例，确保该类的唯一性。
- https://blog.csdn.net/unonoi/article/details/121138176?ops_request_misc=%257B%2522request%255Fid%2522%253A%25223f81a9de7fd6f315cb874871600f0400%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=3f81a9de7fd6f315cb874871600f0400&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-1-121138176-null-null.142^v102^pc_search_result_base1&utm_term=c%2B%2B%E5%8D%95%E4%BE%8B%E6%A8%A1%E5%BC%8F&spm=1018.2226.3001.4187
### 为什么需要单例模式
两个原因：

节省资源。一个类只有一个实例，不存在多份实例，节省资源。
方便控制。在一些操作公共资源的场景时，避免了多个对象引起的复杂操作。
但是在实现单例模式时，需要考虑到线程安全的问题

### 单例类的特点
*   构造函数和析构函数为私有类型，目的是禁止外部构造和析构。
*   拷贝构造函数和赋值构造函数是私有类型，目的是禁止外部拷贝和赋值，确保实例的唯一性。
*   类中有一个获取实例的静态方法，可以全局访问。
*   

## 工厂模式



1\. 引入
------

        我们先不看定义，请各位思考一个问题，我们刚入门c++的时候要创建一个对象该怎么创建？看下面伪代码。

```c++
    class A {
    public:
        xxxx
    private:
        xxxx
    };
    
    class B {
    public:
        xxxx
    private:
        xxxx
    };
    
    int main() {
    // 创建对象a
    A a;
    
    // 创建对象b
    B b;
    }
```

        现在是创建了两个类的对象，那如果有很多类呢？如果项目中需要创建很多对象呢？所以，工厂模式就提供了一种创建对象的方法。我的理解就是把创建对象的过程放到一个工厂类里面去，这个工厂类就相当于一个工厂一样，专门去产生（创建）一个个的产品（对象）。

2\. 三种工厂模式
----------

        我们还是不看定义，接下来我将采用UML类图加代码的形式讲述工厂模式的三种模式，分别是[简单工厂模式](https://so.csdn.net/so/search?q=%E7%AE%80%E5%8D%95%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F&spm=1001.2101.3001.7020)、工厂方法模式、抽象工厂模式。

### 2.1 简单工厂模式

*   场景：假如有一个制造车的工厂，要造SUV， 跑车，MPV三种车。
*   UML图如下所示

![](https://i-blog.csdnimg.cn/direct/1e2994c9b0204baaae90f83b7bcc2fc5.png)

*    根据UML图看如下代码

```c++
#include <iostream>
 
// 汽车抽象类
class Cars
{
public:
    virtual ~Cars() {} // 虚析构函数，确保正确调用派生类的析构函数
    virtual void cars_info() = 0; // 纯虚函数，要求派生类实现
};
 
// SUV汽车
class SUVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是SUV汽车，我的特点是：空间大，适合家庭出行" << std::endl;
    }
};
 
// 运动型汽车
class SportCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是运动型汽车，我的特点是：速度快，外观时尚" << std::endl;
    }
};
 
// MPV汽车
class MPVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是MPV汽车，我的特点是：多功能，适合多人出行" << std::endl;
    }
};
 
// 简单工厂类
class CarsFactory
{
public:
    enum CarsType
    {
        SUV,
        SPORT,
        MPV
    };
 
    static Cars* create_cars(CarsType type)
    {
        switch (type)
        {
        case SUV:
            return new SUVCars();
        case SPORT:
            return new SportCars();
        case MPV:
            return new MPVCars();
        default:
            return nullptr;
        }
    }
};
 
int main()
{
    // 创建SUV汽车
    Cars* suv = CarsFactory::create_cars(CarsFactory::SUV);
    suv->cars_info();
    delete suv;
 
    // 创建运动型汽车
    Cars* sport = CarsFactory::create_cars(CarsFactory::SPORT);
    sport->cars_info();
    delete sport;
 
    // 创建MPV汽车
    Cars* mpv = CarsFactory::create_cars(CarsFactory::MPV);
    mpv->cars_info();
    delete mpv;
 
    return 0;
}

```

        是不是看的一头雾水，别慌我来为你逐个解释，首先既然工厂模式是用一个工厂类去创建其他的对象，我们肯定需要一个工厂类（Carsfactory），那么如何创建呢？我们只需要封装一个creat_cars函数，只要给这个函数一个参数，就可以根据这个参数创建不同的对象。

```c++
// 简单工厂类
class CarsFactory
{
public:
    enum CarsType
    {
        SUV,
        SPORT,
        MPV
    };
 
    static Cars* create_cars(CarsType type)
    {
        switch (type)
        {
        case SUV:
            return new SUVCars();
        case SPORT:
            return new SportCars();
        case MPV:
            return new MPVCars();
        default:
            return nullptr;
        }
    }
};

```

        那么既然创建对象，肯定要有相应的类，所以我们定义了SUVCars, SportCars, MPVCars。
```c++
// SUV汽车
class SUVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是SUV汽车，我的特点是：空间大，适合家庭出行" << std::endl;
    }
};
 
// 运动型汽车
class SportCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是运动型汽车，我的特点是：速度快，外观时尚" << std::endl;
    }
};
 
// MPV汽车
class MPVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是MPV汽车，我的特点是：多功能，适合多人出行" << std::endl;
    }
};

```

         读到这里可能还有小白有疑问，那为什么还要继承抽象类(Cars)呢，直接通过工厂类(Carsfactory)创建对象不就好了，非也。

        这里之所以继承抽象类(Cars)，首先可以通过基类指针或引用来操作不同类型的派生类对象。在这个例子中，我们可以通过Cars\*指针来调用cars_info方法，而不需要关心具体的汽车类型。这使得代码更加灵活和可扩展。

        基类Cars使得工厂类CarsFactory可以返回不同类型的汽车对象，而不需要知道具体的汽车类型。工厂类只需要返回基类指针，这使得工厂类的实现更加简洁和通用。

```c++
// 汽车抽象类
class Cars
{
public:
    virtual ~Cars() {} // 虚析构函数，确保正确调用派生类的析构函数
    virtual void cars_info() = 0; // 纯虚函数，要求派生类实现
};

```

        然后再看main函数，估计就会恍然大悟。

```c++
int main()
{
    // 创建SUV汽车
    Cars* suv = CarsFactory::create_cars(CarsFactory::SUV);
    suv->cars_info();
    delete suv;
 
    // 创建运动型汽车
    Cars* sport = CarsFactory::create_cars(CarsFactory::SPORT);
    sport->cars_info();
    delete sport;
 
    // 创建MPV汽车
    Cars* mpv = CarsFactory::create_cars(CarsFactory::MPV);
    mpv->cars_info();
    delete mpv;
 
    return 0;
}

```

*   看懂了UML图和代码之后，我们来看简单工厂模式的结构组成

        **抽象产品类**： 定义了产品的接口，每个具体产品都要实现这个接口。例如，在我们的例子中，`Cars` 是抽象产品。

        **具体产品类**： 实现了抽象产品接口的具体类。例如，`SUVCars`、`SportCars`、`MPVCars` 是 `Cars` 的具体产品。

        **工厂类**： 核心类，包含一个方法，根据传入的参数决定创建哪种类型的对象。例如，`CarsFactory` 是工厂类，它声明了 `create_cars` 方法。

### 2.2 [工厂方法模式](https://so.csdn.net/so/search?q=%E5%B7%A5%E5%8E%82%E6%96%B9%E6%B3%95%E6%A8%A1%E5%BC%8F&spm=1001.2101.3001.7020)

*   简单工厂模式的问题：

        有了上面对简单工厂的理解，试想这样一种情景：当我们想要添加新的产品类，我们需要直接修改工厂类（CarsFactory），这显然违背了开闭原则（对扩展开放，对修改关闭）。那该怎么办？

        我们可以**从工厂划分出子工厂，把创建对象的任务交给子工厂**不就行了！直接看UML图和代码。

*   UML图如下所示

![](https://i-blog.csdnimg.cn/direct/287d7cf6939d49369147cf84fb38a7db.png)

*   根据UML图看如下代码

```c++
#include <iostream>
 
// 汽车抽象类
class Cars
{
public:
    virtual ~Cars() {} // 虚析构函数，确保正确调用派生类的析构函数
    virtual void cars_info() = 0; // 纯虚函数，要求派生类实现
};
 
// SUV汽车
class SUVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是SUV汽车，我的特点是：空间大，适合家庭出行" << std::endl;
    }
};
 
// 运动型汽车
class SportCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是运动型汽车，我的特点是：速度快，外观时尚" << std::endl;
    }
};
 
// MPV汽车
class MPVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是MPV汽车，我的特点是：多功能，适合多人出行" << std::endl;
    }
};
 
// 抽象工厂类
class CarsFactory
{
public:
    virtual ~CarsFactory() {}
    virtual Cars* create_cars() = 0; // 纯虚函数，要求派生类实现
};
 
// SUV汽车工厂
class SUVCarsFactory : public CarsFactory
{
public:
    Cars* create_cars() override
    {
        return new SUVCars();
    }
};
 
// 运动型汽车工厂
class SportCarsFactory : public CarsFactory
{
public:
    Cars* create_cars() override
    {
        return new SportCars();
    }
};
 
// MPV汽车工厂
class MPVCarsFactory : public CarsFactory
{
public:
    Cars* create_cars() override
    {
        return new MPVCars();
    }
};
 
int main()
{
    // 创建SUV汽车
    CarsFactory* suvFactory = new SUVCarsFactory();
    Cars* suv = suvFactory->create_cars();
    suv->cars_info();
    delete suv;
    delete suvFactory;
 
    // 创建运动型汽车
    CarsFactory* sportFactory = new SportCarsFactory();
    Cars* sport = sportFactory->create_cars();
    sport->cars_info();
    delete sport;
    delete sportFactory;
 
    // 创建MPV汽车
    CarsFactory* mpvFactory = new MPVCarsFactory();
    Cars* mpv = mpvFactory->create_cars();
    mpv->cars_info();
    delete mpv;
    delete mpvFactory;
 
    return 0;
}

```

我想通过我上面的讲解，这里不用细讲，也能看懂了吧？总结一句话就是，具体产品对象的创建用具体的工厂子类完成。

*   再看工厂方法模式的结构，一目了然

        **抽象产品类**： 定义了产品的接口，每个具体产品都要实现这个接口。例如，在我们的例子中，`Cars` 是抽象产品。

        **具体产品类**： 实现了抽象产品接口的具体类。例如，`SUVCars`、`SportCars`、`MPVCars` 是 `Cars` 的具体产品。

        **抽象工厂类**： 声明了一个创建产品的方法。例如，`CarsFactory` 是抽象工厂，它声明了 `create_cars` 方法。

        **具体工厂类**： 实现了抽象工厂接口，负责创建具体产品的实例。例如，`SUVCarsFactory`、`SportCarsFactory`、`MPVCarsFactory` 是具体工厂，它们实现了 `create_cars` 方法，分别创建相应的具体产品。

### 2.3 [抽象工厂模式](https://so.csdn.net/so/search?q=%E6%8A%BD%E8%B1%A1%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F&spm=1001.2101.3001.7020)

        看懂了工厂方法模式，有时候这个工厂啊不仅仅生产一种产品，比如SUV工厂（SUVFactory），它除了产生SUV汽车产品（SUVCars），它有可能还需要生产SUV轮胎，SUV玻璃等等。这就是抽象工厂的内涵，话不多说，这次我相信直接看UML和代码就能看懂。

*   UML图如下所示

![](https://i-blog.csdnimg.cn/direct/2e5b987094e4480e8d0d7d804db0b405.png)

*   根据UML图看代码（我建议直接从main函数开始看）
```c++
#include <iostream>
 
// 汽车抽象类
class Cars
{
public:
    virtual ~Cars() {} // 虚析构函数，确保正确调用派生类的析构函数
    virtual void cars_info() = 0; // 纯虚函数，要求派生类实现
};
 
// 轮胎抽象类
class Tires
{
public:
    virtual ~Tires() {}
    virtual void tire_info() = 0;
};
 
// SUV汽车
class SUVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是SUV汽车，我的特点是：空间大，适合家庭出行" << std::endl;
    }
};
 
// SUV轮胎
class SUVTires : public Tires
{
public:
    void tire_info() override
    {
        std::cout << "我是SUV轮胎，适合各种地形" << std::endl;
    }
};
 
// 运动型汽车
class SportCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是运动型汽车，我的特点是：速度快，外观时尚" << std::endl;
    }
};
 
// 运动型轮胎
class SportTires : public Tires
{
public:
    void tire_info() override
    {
        std::cout << "我是运动型轮胎，适合高速行驶" << std::endl;
    }
};
 
// MPV汽车
class MPVCars : public Cars
{
public:
    void cars_info() override
    {
        std::cout << "我是MPV汽车，我的特点是：多功能，适合多人出行" << std::endl;
    }
};
 
// MPV轮胎
class MPVTires : public Tires
{
public:
    void tire_info() override
    {
        std::cout << "我是MPV轮胎，适合长途旅行" << std::endl;
    }
};
 
// 抽象工厂类
class CarsFactory
{
public:
    virtual ~CarsFactory() {}
    virtual Cars* create_cars() = 0; // 纯虚函数，要求派生类实现
    virtual Tires* create_tires() = 0; // 纯虚函数，要求派生类实现
};
 
// SUV工厂
class SUVCarsFactory : public CarsFactory
{
public:
    Cars* create_cars() override
    {
        return new SUVCars();
    }
 
    Tires* create_tires() override
    {
        return new SUVTires();
    }
};
 
// 运动型汽车工厂
class SportCarsFactory : public CarsFactory
{
public:
    Cars* create_cars() override
    {
        return new SportCars();
    }
 
    Tires* create_tires() override
    {
        return new SportTires();
    }
};
 
// MPV工厂
class MPVCarsFactory : public CarsFactory
{
public:
    Cars* create_cars() override
    {
        return new MPVCars();
    }
 
    Tires* create_tires() override
    {
        return new MPVTires();
    }
};
 
int main()
{
    // 创建SUV汽车和轮胎
    CarsFactory* suvFactory = new SUVCarsFactory();
    Cars* suv = suvFactory->create_cars();
    Tires* suvTires = suvFactory->create_tires();
    suv->cars_info();
    suvTires->tire_info();
    delete suv;
    delete suvTires;
    delete suvFactory;
 
    // 创建运动型汽车和轮胎
    CarsFactory* sportFactory = new SportCarsFactory();
    Cars* sport = sportFactory->create_cars();
    Tires* sportTires = sportFactory->create_tires();
    sport->cars_info();
    sportTires->tire_info();
    delete sport;
    delete sportTires;
    delete sportFactory;
 
    // 创建MPV汽车和轮胎
    CarsFactory* mpvFactory = new MPVCarsFactory();
    Cars* mpv = mpvFactory->create_cars();
    Tires* mpvTires = mpvFactory->create_tires();
    mpv->cars_info();
    mpvTires->tire_info();
    delete mpv;
    delete mpvTires;
    delete mpvFactory;
 
    return 0;
}
```

        是不是看到这么宽的UML图，和这么长的代码就感觉不好理解。有什么不好理解的，直接看main函数，你会发现其实SUV工厂（SUVFactory）造SUV车产品（SUVCars）和SUV轮胎产品（SUVTires）

        **总结就是一个具体工厂类，可以造多个同系列的产品。**

        还难吗？？

*   抽象工厂模式的结构

        **抽象产品类**：定义了产品的接口，每个具体产品都要实现这个接口。例如，在我们的例子中，`Cars` 和 `Tires` 是抽象产品。

        **具体产品类**：实现了抽象产品接口的具体类。例如，`SUVCars`、`SportCars`、`MPVCars` 是 `Cars` 的具体产品，`SUVTires`、`SportTires`、`MPVTires` 是 `Tires` 的具体产品

        **抽象工厂类**：声明了一组创建产品的方法，每个方法对应一个抽象产品。例如，`CarsFactory` 是抽象工厂，它声明了 `create_cars` 和 `create_tires` 方法。

        **具体工厂类**：实现了抽象工厂接口，负责创建具体产品的实例。例如，`SUVCarsFactory`、`SportCarsFactory`、`MPVCarsFactory` 是具体工厂，它们实现了 `create_cars` 和 `create_tires` 方法，分别创建相应的具体产品。

3\. 工厂模式的定义
-----------

        有了上面的讲解，我们现在看工厂模式的定义，是不是就很清楚了。

        工厂模式（Factory Pattern）是一种创建型设计模式，它提供了一种创建对象的方式，使得创建对象的过程与使用对象的过程分离。工厂模式的主要目的是将对象的创建过程封装在工厂类中，客户端代码只需要关心从工厂获取对象的过程，而不需要了解对象的创建细节。

参考资料
----

*   [C++ 深入浅出工厂模式（初识篇） - 小林coding - 博客园 (cnblogs.com)](https://www.cnblogs.com/xiaolincoding/p/11524376.html "C++ 深入浅出工厂模式（初识篇） - 小林coding - 博客园 (cnblogs.com)")
*   [工厂模式 | 菜鸟教程 (runoob.com)](https://www.runoob.com/design-pattern/factory-pattern.html "工厂模式 | 菜鸟教程 (runoob.com)")







