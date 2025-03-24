#### 类型转换


一、C语言中的类型转换
-----------

> C语言和C++都是强类型语言，如果[赋值运算符](https://so.csdn.net/so/search?q=%E8%B5%8B%E5%80%BC%E8%BF%90%E7%AE%97%E7%AC%A6&spm=1001.2101.3001.7020)左右两侧变量的类型不同，或形参与实参的类型不匹配，或返回值类型与接收返回值的变量类型不一致，那么就需要进行类型转换。

那么C语言有两种类型转换，分别是显示类型转换和[隐式类型转换](https://so.csdn.net/so/search?q=%E9%9A%90%E5%BC%8F%E7%B1%BB%E5%9E%8B%E8%BD%AC%E6%8D%A2&spm=1001.2101.3001.7020)，我们看下面的介绍：

*   **隐式类型转换**：编译器在编译阶段自动进行，能转就转，不能转就编译失败。
*   **显式类型转换**：需要用户自己处理，以(指定类型)变量的方式进行类型转换。

需要注意的是，只有相近类型之间才能发生隐式类型转换，比如int和double表示的都是数值，只不过它们表示的范围和精度不同。而指针类型表示的是地址编号，因此整型和指针类型之间不会进行隐式类型转换，如果需要转换则只能进行显式类型转换。

    int main()
    {
    	// 隐式类型转换
    	int i = 0;
    	double d = i;
    	std::cout << i << std::endl;
    	std::cout << d << std::endl;
    	// 显示类型转换
    	int* p = &i;
    	int address = (int)p;
    	cout << p << endl;
    	cout << address << endl;
    	return 0;
    }
    

![](https://i-blog.csdnimg.cn/blog_migrate/78b7434c18cf2433da54f46b6b35222d.png)

二、C++的四种类型转换及其原因
----------------

C语言的两种类型转换是很有用且对于我们的代码来讲确实是一个很大的进步，而有优点则必定是有缺陷的，缺陷如下：

*   隐式类型转换在某些情况下可能会出问题，比如**数据精度丢失**。
*   显式类型转换将所有情况混合在一起，**转换的可视性比较差**。

因此C++为了加强类型转换的可视性，引入了四种命名的强制类型转换操作符，分别是**static\_cast、reinterpret\_cast、const\_cast和dynamic\_cast**。我们一一进行介绍。

### 1、C++强制类型转换

#### （1）static\_cast

static\_cast用于相近类型之间的转换，编译器隐式执行的任何类型转换都可用static\_cast，但它不能用于两个不相关类型之间转换。

    int main()
    {
    	double d = 10;
    	int i = static_cast<int>(d);
    	std::cout << i << std::endl;
    
    	int* p = &i;
    	// int x = static_cast<int> p; // error
    	return 0;
    }
    

#### （2）reinterpret\_cast

reinterpret\_cast用于两个不相关类型之间的转换。

    int main()
    {
    	int i = 10;
    	int* p = &i;
    	int x = reinterpret_cast<int>(p);
    	std::cout << x << std::endl;
    	return 0;
    }
    

![](https://i-blog.csdnimg.cn/blog_migrate/b1c1e38dcbfba826e3d331b6a2a08934.png)

小知识：在下面的代码中将带参带返回值的函数指针转换成了无参无返回值的函数指针，并且还可以用转换后函数指针调用这个函数。

    typedef void(*FUNC)();
    int DoSomething(int i)
    {
    	cout << "DoSomething: " << i << endl;
    	return 0;
    }
    int main()
    {
    	FUNC f = reinterpret_cast<FUNC>(DoSomething);
    	f();
    	return 0;
    }
    

用转换后的函数指针调用该函数时没有传入参数，因此这里打印出参数i的值是一个随机值。

#### （3）const\_cast

const\_cast用于删除变量的const属性，转换后就可以对const变量的值进行修改。

    int main()
    {
    	const int a = 10;
    	int* p = const_cast<int*>(&a);
    	*p = 20;
    	std::cout << a << std::endl;
    	std::cout << *p << std::endl;
    	return 0;
    }
    

![](https://i-blog.csdnimg.cn/blog_migrate/e6e5902f793138ffb37b0f76bf7b42e2.png)

*   代码中用const\_cast删除了变量a的地址的const属性，这时就可以通过这个指针来修改变量a的值。
*   由于编译器认为const修饰的变量是不会被修改的，因此会将const修饰的变量存放到寄存器当中，当需要读取const变量时就会直接从寄存器中进行读取，而我们修改的实际上是内存中的a的值，因此最终打印出a的值是未修改之前的值。
*   如果不想让编译器将const变量优化到寄存器当中，可以用volatile关键字对const变量进行修饰，这时当要读取这个const变量时编译器就会从内存中进行读取，即保持了该变量在内存中的可见性。

#### （4）dynamic\_cast

dynamic\_cast用于将父类的指针（或引用）转换成子类的指针（或引用）。

##### i、向上转型与向下转型

**向上转型**： 子类的指针（或引用）→ 父类的指针（或引用）。  
**向下转型**： 父类的指针（或引用）→ 子类的指针（或引用）。

其中，**向上转型就是所说的切割/切片**，是语法天然支持的，不需要进行转换，而**向下转型是语法不支持的，需要进行强制类型转换**。

##### ii、向下转型的安全问题

向下转换分为两种情况：

1.  如果**父类的指针（或引用）指向的是一个父类对象**，那么将其转换为子类的指针（或引用）是不安全的，因为转换后可能会访问到子类的资源，而这个资源是父类对象所没有的。
2.  如果**父类的指针（或引用）指向的是一个子类对象**，那么将其转换为子类的指针（或引用）则是安全的。

使用dynamic\_cast进行向下转型则是安全的，如果父类的指针（或引用）指向的是子类对象那么dynamic\_cast会转换成功，但如果父类的指针（或引用）指向的是父类对象那么dynamic\_cast会转换失败并返回一个空指针。

    class A
    {
    public:
    	virtual void f()
    	{}
    };
    class B : public A
    {};
    void func(A* pa)
    {
    	B* pb1 = (B*)pa;               //不安全
    	B* pb2 = dynamic_cast<B*>(pa); //安全
    
    	cout << "pb1: " << pb1 << endl;
    	cout << "pb2: " << pb2 << endl;
    }
    int main()
    {
    	A a;
    	B b;
    	func(&a);
    	func(&b);
    	return 0;
    }
    

上述代码中，如果传入func函数的是子类对象的地址，那么在转换后pb1和pb2都会有对应的地址，但如果传入func函数的是父类对象的地址，那么转换后pb1会有对应的地址，而pb2则是一个空指针。  
**说明一下**： dynamic\_cast只能用于含有虚函数的类，因为运行时类型检查需要运行时的类型信息，而这个信息是存储在虚函数表中的，只有定义了虚函数的类才有虚函数表。

### 2、explicit

explicit用来修饰构造函数，从而禁止单参数构造函数的隐式转换。

    class A
    {
    private:
    	int _a;
    public:
    	explicit A(int a)
    	{
    		std::cout << "explicit A(int a) " << std::endl;
    	}
    	A(const A& a)
    	{
    		std::cout << "A(const A& a) " << std::endl;
    	}
    };
    
    int main()
    {
    	A a1(1);
    	// A a2 = 2; // error 
    	return 0;
    }
    

语法上，`A a2 = 2;`等价于

    A tmp(1); // 先构建临时tmp
    A a2(tmp); // 再拷贝构造
    

所以在早期的编译器中，当编译器遇到`A a2 = 1`这句代码时，会先构造一个临时对象，再用这个临时对象拷贝构造a2。但是现在的编译器已经做了优化，当遇到`A a2 = 1`这句代码时，会直接按照`A a2(1)`的方式进行处理，这也叫做隐式类型转换。  
但对于单参数的自定义类型来说，`A a2 = 1`这种代码的可读性不是很好，因此可以用explicit修饰单参数的构造函数，从而禁止单参数构造函数的隐式转换。

二、RTTI
------

RTTI（Run-Time Type Identification）就是运行时类型识别。

C++通过以下三种形式支持 RTTI：

1.  typeid：在运行时识别出一个对象的类型。
2.  dynamic\_cast：在运行时识别出一个父类的指针（或引用）指向的是父类对象还是子类对象。
3.  decltype：在运行时推演出一个表达式或函数返回值的类型。