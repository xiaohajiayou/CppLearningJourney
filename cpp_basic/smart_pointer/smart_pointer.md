

一、为什么要使用智能指针
------------

一句话带过：智能指针就是帮我们C++程序员管理动态分配的内存的，它会帮助我们自动释放new出来的内存，从而**避免内

如下例子就是内存泄露的例子：

    #include <iostream>
    #include <string>
    #include <memory>
    
    using namespace std;
    
    
    // 动态分配内存，没有释放就return
    void memoryLeak1() {
    	string *str = new string("动态分配内存！");
    	return;
    }
    
    // 动态分配内存，虽然有些释放内存的代码，但是被半路截胡return了
    int memoryLeak2() {
    	string *str = new string("内存泄露！");
    
    	// ...此处省略一万行代码
    
    	// 发生某些异常，需要结束函数
    	if (1) {
    		return -1;
    	}
    	/
    	// 另外，使用try、catch结束函数，也会造成内存泄漏！
    	/
    
    	delete str;	// 虽然写了释放内存的代码，但是遭到函数中段返回，使得指针没有得到释放
    	return 1;
    }
    
    
    int main(void) {
    
    	memoryLeak1();
    
    	memoryLeak2();
    
    	return 0;
    } 
    

memoryLeak1函数中，new了一个[字符串指针](https://so.csdn.net/so/search?q=%E5%AD%97%E7%AC%A6%E4%B8%B2%E6%8C%87%E9%92%88&spm=1001.2101.3001.7020)，但是没有delete就已经return结束函数了，导致内存没有被释放，内存泄露！  
memoryLeak2函数中，new了一个字符串指针，虽然在函数末尾有些释放内存的代码delete str，但是在delete之前就已经return了，所以内存也没有被释放，内存泄露！

使用指针，我们没有释放，就会造成内存泄露。但是我们使用普通对象却不会！

**思考**：如果我们分配的动态内存都交由有生命周期的对象来处理，那么在对象过期时，让它的析构函数删除指向的内存，这看似是一个 very nice 的方案？

智能指针就是通过这个原理来解决指针自动释放的问题！

1.  C++98 提供了 auto_ptr 模板的解决方案
2.  C++11 增加unique_ptr、shared_ptr 和weak_ptr

* * *

二、auto_ptr
-----------

auto_ptr 是c++ 98定义的智能指针模板，其定义了管理指针的对象，可以将new 获得（直接或间接）的地址赋给这种对象。当对象过期时，其析构函数将使用delete 来释放内存！

用法:  
头文件: #include < memory >  
用 法: auto_ptr<类型> 变量名(new 类型)

例 如:  
auto_ptr< string > str(new string(“我要成为大牛~ 变得很牛逼！”));  
auto_ptr<vector< int >> av(new vector< int >());  
auto_ptr< int > array(new int\[10\]);



**使用智能指针**：

    // 定义智能指针
    auto_ptr<Test> test(new Test);
    

智能指针可以像普通指针那样使用：

    cout << "test->debug：" << test->getDebug() << endl;
    cout << "(*test).debug：" << (*test).getDebug() << endl;
    

这时再试试：

    int main(void) {
    
    	//Test *test = new Test;
    	auto_ptr<Test> test(new Test);
    
    	cout << "test->debug：" << test->getDebug() << endl;
    	cout << "(*test).debug：" << (*test).getDebug() << endl;
    
    	return 0;
    } 
    

![](https://i-blog.csdnimg.cn/blog_migrate/4ce4a69a023707d189e9b1506e5bfc44.png#pic_center)  
自动调用了析构函数。  
**为什么智能指针可以像普通指针那样使用**？？？  
因为其里面重载了 \* 和 -> 运算符， \* 返回普通对象，而 -> 返回指针对象。  
![](https://i-blog.csdnimg.cn/blog_migrate/942935525e8e85d3c1d9dd6c78999bfd.png#pic_center)  
具体原因不用深究，只需知道他为什么可以这样操作就像！  
函数中返回的是调用get()方法返回的值，那么这个get()是什么呢？

智能指针的三个常用函数：

1.  **get**() 获取智能指针托管的指针地址
    
        // 定义智能指针
        auto_ptr<Test> test(new Test);
        
        Test *tmp = test.get();		// 获取指针返回
        cout << "tmp->debug：" << tmp->getDebug() << endl;
        
    
    但我们一般不会这样使用，因为都可以直接使用智能指针去操作，除非有一些特殊情况。  
    **函数原型**：
    
        _NODISCARD _Ty * get() const noexcept
        {	// return wrapped pointer
        	return (_Myptr);
        }
        
    
2.  **release**() 取消智能指针对动态内存的托管
    
        // 定义智能指针
        auto_ptr<Test> test(new Test);
        
        Test *tmp2 = test.release();	// 取消智能指针对动态内存的托管
        delete tmp2;	// 之前分配的内存需要自己手动释放
        
    
    也就是智能指针不再对该指针进行管理，改由管理员进行管理！  
    **函数原型**：
    
        _Ty * release() noexcept
        {	// return wrapped pointer and give up ownership
        	_Ty * _Tmp = _Myptr;
        	_Myptr = nullptr;
        	return (_Tmp);
        }
        
    
3.  **reset**() 重置智能指针托管的内存地址，如果地址不一致，原来的会被析构掉
    
        // 定义智能指针
        auto_ptr<Test> test(new Test);
        
        test.reset();			// 释放掉智能指针托管的指针内存，并将其置NULL
        
        test.reset(new Test());	// 释放掉智能指针托管的指针内存，并将参数指针取代之
        
    
    reset函数会将参数的指针(不指定则为NULL)，与托管的指针比较，如果地址不一致，那么就会析构掉原来托管的指针，然后使用参数的指针替代之。然后智能指针就会托管参数的那个指针了。  
    **函数原型：**
    
        void reset(_Ty * _Ptr = nullptr)
        {	// destroy designated object and store new pointer
        	if (_Ptr != _Myptr)
        		delete _Myptr;
        	_Myptr = _Ptr;
        }
        
    

**使用建议**：

1.  尽可能不要将auto_ptr 变量定义为全局变量或指针；
    
        // 没有意义，全局变量也是一样
        auto_ptr<Test> *tp = new auto_ptr<Test>(new Test);	
        
    
2.  除非自己知道后果，不要把auto_ptr 智能指针赋值给同类型的另外一个 智能指针；
    
        auto_ptr<Test> t1(new Test);
        auto_ptr<Test> t2(new Test);
        t1 = t2;	// 不要这样操作...
        
    
3.  C++11 后auto_ptr 已经被“抛弃”，已使用unique_ptr替代！C++11后不建议使用auto_ptr。
    
4.  **auto_ptr 被C++11抛弃的主要原因**
    
    **1).** 复制或者赋值都会改变资源的所有权
    
        // auto_ptr 被C++11抛弃的主要原因
        auto_ptr<string> p1(new string("I'm Li Ming!"));
        auto_ptr<string> p2(new string("I'm age 22."));
        
        cout << "p1：" << p1.get() << endl;
        cout << "p2：" << p2.get() << endl;
        
        // p2赋值给p1后，首先p1会先将自己原先托管的指针释放掉，然后接收托管p2所托管的指针，
        // 然后p2所托管的指针制NULL，也就是p1托管了p2托管的指针，而p2放弃了托管。
        p1 = p2;	
        cout << "p1 = p2 赋值后：" << endl;
        cout << "p1：" << p1.get() << endl;
        cout << "p2：" << p2.get() << endl;
        
    
    ![](https://i-blog.csdnimg.cn/blog_migrate/18102893b9f701b1afd73b5d244a2b39.png#pic_center)
    
    **2).** 在STL容器中使用auto_ptr存在着重大风险，因为容器内的元素必须支持可复制和可赋值
    
        vector<auto_ptr<string>> vec;
        auto_ptr<string> p3(new string("I'm P3"));
        auto_ptr<string> p4(new string("I'm P4"));
        
        // 必须使用std::move修饰成右值，才可以进行插入容器中
        vec.push_back(std::move(p3));
        vec.push_back(std::move(p4));
        
        cout << "vec.at(0)：" <<  *vec.at(0) << endl;
        cout << "vec[1]：" <<  *vec[1] << endl;
        
        
        // 风险来了：
        vec[0] = vec[1];	// 如果进行赋值，问题又回到了上面一个问题中。
        cout << "vec.at(0)：" << *vec.at(0) << endl;
        cout << "vec[1]：" << *vec[1] << endl;
        
    
    访问越界了！  
    ![](https://i-blog.csdnimg.cn/blog_migrate/de1eefaaaf0efd1c5eee484e298f34dd.png#pic_center)
    
    **3).** 不支持对象数组的内存管理
    
        auto_ptr<int[]> array(new int[5]);	// 不能这样定义
        
    
    ![](https://i-blog.csdnimg.cn/blog_migrate/2807fde64f58f77dc37843e7ae735dc7.png#pic_center)
    

所以，C++11用更严谨的unique_ptr 取代了auto_ptr！

**测试代码**：

    #include <iostream>
    #include <string>
    #include <memory>
    #include <vector>
    
    using namespace std;
    
    class Test {
    public:
    	Test() { cout << "Test的构造函数..." << endl; }
    	~Test() { cout << "Test的析构函数..." << endl; }
    
    	int getDebug() { return this->debug; }
    
    private:
    	int debug = 20;
    };
    
    // 不要定义为全局变量，没有意义
    //auto_ptr<Test> test(new Test);
    
    void memoryLeak1() {
    	//Test *test = new Test;
    
    	// 定义智能指针
    	auto_ptr<Test> test(new Test);
    	
    	cout << "test->debug：" << test->getDebug() << endl;
    	cout << "(*test).debug：" << (*test).getDebug() << endl;
    
    
    	// get方法
    	Test *tmp = test.get();		// 获取指针返回
    	cout << "tmp->debug：" << tmp->getDebug() << endl;
    
    
    	// release方法
    	Test *tmp2 = test.release();	// 取消智能指针对动态内存的托管
    	delete tmp2;	// 之前分配的内存需要自己手动释放
    
    
    	// reset方法：重置智能指针托管的内存地址，如果地址不一致，原来的会被析构掉
    	test.reset();			// 释放掉智能指针托管的指针内存，并将其置NULL
    	test.reset(new Test());	// 释放掉智能指针托管的指针内存，并将参数指针取代之
    
    
    	// 忠告：不要将智能指针定义为指针
    	//auto_ptr<Test> *tp = new auto_ptr<Test>(new Test);
    
    	// 忠告：不要定义指向智能指针对象的指针变量
    	//auto_ptr<Test> t1(new Test);
    	//auto_ptr<Test> t2(new Test);
    	//t1 = t2;
    
    	return;
    }
    
    int memoryLeak2() {
    	//Test *test = new Test();
    
    	// 定义智能指针
    	auto_ptr<Test> test(new Test);
    
    	// ...此处省略一万行代码
    
    	// 发生某些异常，需要结束函数
    	if (1) {
    		return -1;
    	}
    
    	//delete test;
    	return 1;
    }
    
    
    int main1(void) {
    
    	//memoryLeak1();
    
    	//memoryLeak2();
    
    	//Test *test = new Test;
    	//auto_ptr<Test> test(new Test);
    
    	//cout << "test->debug：" << test->getDebug() << endl;
    	//cout << "(*test).debug：" << (*test).getDebug() << endl;
    
    
    	 auto_ptr 被C++11抛弃的主要原因
    	//auto_ptr<string> p1(new string("I'm Li Ming!"));
    	//auto_ptr<string> p2(new string("I'm age 22."));
    	//
    	//cout << "p1：" << p1.get() << endl;
    	//cout << "p2：" << p2.get() << endl;
    
    	//p1 = p2;
    	//cout << "p1 = p2 赋值后：" << endl;
    	//cout << "p1：" << p1.get() << endl;
    	//cout << "p2：" << p2.get() << endl;
    
    
    
    	// 弊端2.在STL容器中使用auto_ptr存在着重大风险，因为容器内的元素必须支持可复制
    	vector<auto_ptr<string>> vec;
    	auto_ptr<string> p3(new string("I'm P3"));
    	auto_ptr<string> p4(new string("I'm P4"));
    
    	vec.push_back(std::move(p3));
    	vec.push_back(std::move(p4));
    
    	cout << "vec.at(0)：" <<  *vec.at(0) << endl;
    	cout << "vec[1]：" <<  *vec[1] << endl;
    
    
    	// 风险来了：
    	vec[0] = vec[1];
    	cout << "vec.at(0)：" << *vec.at(0) << endl;
    	cout << "vec[1]：" << *vec[1] << endl;
    
    
    	// 弊端3.不支持对象数组的内存管理
    	//auto_ptr<int[]> array(new int[5]);	// 不能这样定义
    	return 0;
    } 
    

* * *

三、unique_ptr
-------------

auto_ptr是用于C++11之前的智能指针。由于 auto_ptr 基于排他所有权模式：两个指针不能指向同一个资源，复制或赋值都会改变资源的所有权。auto_ptr 主要有三大问题：

1.  复制和赋值会改变资源的所有权，不符合人的直觉。
2.  在 STL 容器中使用auto_ptr存在重大风险，因为容器内的元素必需支持可复制（copy constructable）和可赋值（assignable）。
3.  不支持对象数组的操作

以上问题已经在上面体现出来了，下面将使用unique_ptr解决这些问题。

所以，C++11用更严谨的unique_ptr 取代了auto_ptr！

unique_ptr 和 auto_ptr用法几乎一样，除了一些特殊。

**unique_ptr特性**

1.  基于排他所有权模式：两个指针不能指向同一个资源
2.  无法进行左值unique_ptr复制构造，也无法进行左值复制赋值操作，但允许临时右值赋值构造和赋值
3.  保存指向某个对象的指针，当它本身离开作用域时会自动释放它指向的对象。
4.  在容器中保存指针是安全的

**A**. 无法进行左值复制赋值操作，但允许临时右值赋值构造和赋值

    unique_ptr<string> p1(new string("I'm Li Ming!"));
    unique_ptr<string> p2(new string("I'm age 22."));
    	
    cout << "p1：" << p1.get() << endl;
    cout << "p2：" << p2.get() << endl;
    
    p1 = p2;					// 禁止左值赋值
    unique_ptr<string> p3(p2);	// 禁止左值赋值构造
    
    unique_ptr<string> p3(std::move(p1));
    p1 = std::move(p2);	// 使用move把左值转成右值就可以赋值了，效果和auto_ptr赋值一样
    
    cout << "p1 = p2 赋值后：" << endl;
    cout << "p1：" << p1.get() << endl;
    cout << "p2：" << p2.get() << endl;
    

![](https://i-blog.csdnimg.cn/blog_migrate/4643d79cfaf29757de6e04ce59b06bbf.png#pic_center)

运行截图：  
![](https://i-blog.csdnimg.cn/blog_migrate/019d9931797fb12d51d1ac8a2d99c78b.png#pic_center)

**B**. 在 STL 容器中使用unique_ptr，不允许直接赋值

    vector<unique_ptr<string>> vec;
    unique_ptr<string> p3(new string("I'm P3"));
    unique_ptr<string> p4(new string("I'm P4"));
    
    vec.push_back(std::move(p3));
    vec.push_back(std::move(p4));
    
    cout << "vec.at(0)：" << *vec.at(0) << endl;
    cout << "vec[1]：" << *vec[1] << endl;
    
    vec[0] = vec[1];	/* 不允许直接赋值 */
    vec[0] = std::move(vec[1]);		// 需要使用move修饰，使得程序员知道后果
    
    cout << "vec.at(0)：" << *vec.at(0) << endl;
    cout << "vec[1]：" << *vec[1] << endl;
    

![](https://i-blog.csdnimg.cn/blog_migrate/a8a18955af23da5018857f06e76771da.png#pic_center)

当然，运行后是直接报错的，因为vec\[1\]已经是NULL了，再继续访问就越界了。

**C**. 支持对象数组的内存管理

    // 会自动调用delete [] 函数去释放内存
    unique_ptr<int[]> array(new int[5]);	// 支持这样定义
    

除了上面ABC三项外，unique_ptr的其余用法都与auto_ptr用法一致。

    

### auto_ptr 与 unique_ptr智能指针的内存管理陷阱

    auto_ptr<string> p1;
    string *str = new string("智能指针的内存管理陷阱");
    p1.reset(str);	// p1托管str指针
    {
    	auto_ptr<string> p2;
    	p2.reset(str);	// p2接管str指针时，会先取消p1的托管，然后再对str的托管
    }
    
    // 此时p1已经没有托管内容指针了，为NULL，在使用它就会内存报错！
    cout << "str：" << *p1 << endl;
    

![](https://i-blog.csdnimg.cn/blog_migrate/1f6a4d5d02444d5e5fc7331e02042adc.png#pic_center)  
这是由于auto_ptr 与 unique_ptr的排他性所导致的！  
**为了解决这样的问题，我们可以使用shared_ptr指针指针！**

* * *

四、shared_ptr
-------------

熟悉了unique_ptr 后，其实我们发现unique_ptr 这种排他型的内存管理并不能适应所有情况，有很大的局限！如果需要多个指针变量共享怎么办？

如果有一种方式，可以记录引用特定内存对象的智能指针数量，当复制或拷贝时，**引用计数**加1，当智能指针析构时，**引用计数**减1，如果计数为零，代表已经没有指针指向这块内存，那么我们就释放它！这就是 shared_ptr 采用的策略！

![](https://i-blog.csdnimg.cn/blog_migrate/edae7e643a850589e8825ea3f813a8a1.png#pic_center)



1.  **引用计数的使用**
    
    调用**use_count**函数可以获得当前托管指针的引用计数。
    
        shared_ptr<Person> sp1;
        
        shared_ptr<Person> sp2(new Person(2));
        
        // 获取智能指针管控的共享指针的数量	use_count()：引用计数
        cout << "sp1	use_count() = " << sp1.use_count() << endl;
        cout << "sp2	use_count() = " << sp2.use_count() << endl << endl;
        
        // 共享
        sp1 = sp2;
        
        cout << "sp1	use_count() = " << sp1.use_count() << endl;
        cout << "sp2	use_count() = " << sp2.use_count() << endl << endl;
        
        shared_ptr<Person> sp3(sp1);
        cout << "sp1	use_count() = " << sp1.use_count() << endl;
        cout << "sp2	use_count() = " << sp2.use_count() << endl;
        cout << "sp2	use_count() = " << sp3.use_count() << endl << endl;
        
    
    如上代码，sp1 = sp2; 和 shared_ptr< Person > sp3(sp1);就是在使用引用计数了。
    
    sp1 = sp2; --> sp1和sp2共同托管同一个指针，所以他们的引用计数为2；  
    shared_ptr< Person > sp3(sp1); --> sp1和sp2和sp3共同托管同一个指针，所以他们的引用计数为3；  
    ![](https://i-blog.csdnimg.cn/blog_migrate/ef2e540fe3728d7bbee0a4ea19574520.png#pic_center)
    
2.  **构造**
    
    **1).** shared_ptr< T > sp1; 空的shared_ptr，可以指向类型为T的对象
    
        shared_ptr<Person> sp1;
        Person *person1 = new Person(1);
        sp1.reset(person1);	// 托管person1
        
    
    **2).** shared_ptr< T > sp2(new T()); 定义shared_ptr,同时指向类型为T的对象
    
        shared_ptr<Person> sp2(new Person(2));
        shared_ptr<Person> sp3(sp1);
        
    
    **3).** shared_ptr<T\[\]> sp4; 空的shared_ptr，可以指向类型为T\[\]的数组对象 **C++17后支持**
    
        shared_ptr<Person[]> sp4;
        
    
    **4).** shared_ptr<T\[\]> sp5(new T\[\] { … }); 指向类型为T的数组对象 **C++17后支持**
    
        shared_ptr<Person[]> sp5(new Person[5] { 3, 4, 5, 6, 7 });
        
    
    **5).** shared_ptr< T > sp6(NULL, D()); //空的shared_ptr，接受一个D类型的删除器，使用D释放内存
    
        shared_ptr<Person> sp6(NULL, DestructPerson());
        
    
    **6).** shared_ptr< T > sp7(new T(), D()); //定义shared_ptr,指向类型为T的对象，接受一个D类型的删除器，使用D删除器来释放内存
    
        shared_ptr<Person> sp7(new Person(8), DestructPerson());
        
    
3.  **初始化**
    
    **1).** 方式一：构造函数
    
        shared_ptr<int> up1(new int(10));  // int(10) 的引用计数为1
        shared_ptr<int> up2(up1);  // 使用智能指针up1构造up2, 此时int(10) 引用计数为2
        
    
    **2).** 方式二：使用make_shared 初始化对象，分配内存效率更高(推荐使用)  
    make_shared函数的主要功能是在动态内存中分配一个对象并初始化它，返回指向此对象的shared_ptr; 用法：  
    make_shared<类型>(构造类型对象需要的参数列表);
    
        shared_ptr<int> up3 = make_shared<int>(2); // 多个参数以逗号','隔开，最多接受十个
        shared_ptr<string> up4 = make_shared<string>("字符串");
        shared_ptr<Person> up5 = make_shared<Person>(9);
        
    
4.  **赋值**
    
        shared_ptrr<int> up1(new int(10));  // int(10) 的引用计数为1
        shared_ptr<int> up2(new int(11));   // int(11) 的引用计数为1
        up1 = up2;	// int(10) 的引用计数减1,计数归零内存释放，up2共享int(11)给up1, int(11)的引用计数为2
        
    
5.  **主动释放对象**
    
        shared_ptrr<int> up1(new int(10));
        up1 = nullptr ;	// int(10) 的引用计数减1,计数归零内存释放 
        // 或
        up1 = NULL; // 作用同上 
        
    
6.  **重置**  
    p.reset() ; 将p重置为空指针，所管理对象引用计数 减1  
    p.reset(p1); 将p重置为p1（的值）,p 管控的对象计数减1，p接管对p1指针的管控  
    p.reset(p1,d); 将p重置为p1（的值），p 管控的对象计数减1并使用d作为删除器  
    p1是一个指针！
    
7.  **交换**  
    p1 和 p2 是智能指针
    
        std::swap(p1,p2); // 交换p1 和p2 管理的对象，原对象的引用计数不变
        p1.swap(p2);    // 交换p1 和p2 管理的对象，原对象的引用计数不变
        
    

### shared_ptr使用陷阱

shared_ptr作为被管控的对象的成员时，小心因循环引用造成无法释放资源!

如下代码：  
Boy类中有Girl的智能指针；  
Girl类中有Boy的智能指针；  
当他们交叉互相持有对方的管理对象时…

    #include <iostream>
    #include <string>
    #include <memory>
    
    using namespace std;
    
    class Girl;
    
    class Boy {
    public:
    	Boy() {
    		cout << "Boy 构造函数" << endl;
    	}
    
    	~Boy() {
    		cout << "~Boy 析构函数" << endl;
    	}
    
    	void setGirlFriend(shared_ptr<Girl> _girlFriend) {
    		this->girlFriend = _girlFriend;
    	}
    
    private:
    	shared_ptr<Girl> girlFriend;
    };
    
    class Girl {
    public:
    	Girl() {
    		cout << "Girl 构造函数" << endl;
    	}
    
    	~Girl() {
    		cout << "~Girl 析构函数" << endl;
    	}
    
    	void setBoyFriend(shared_ptr<Boy> _boyFriend) {
    		this->boyFriend = _boyFriend;
    	}
    
    private:
    	shared_ptr<Boy> boyFriend;
    };
    
    
    void useTrap() {
    	shared_ptr<Boy> spBoy(new Boy());
    	shared_ptr<Girl> spGirl(new Girl());
    
    	// 陷阱用法
    	spBoy->setGirlFriend(spGirl);
    	spGirl->setBoyFriend(spBoy);
    	// 此时boy和girl的引用计数都是2
    }
    
    
    int main(void) {
    	useTrap();
    
    	system("pause");
    	return 0;
    }
    

运行截图：

![](https://i-blog.csdnimg.cn/blog_migrate/1fbe1735782ee2a01db524af08eb8d13.png#pic_center)

可以看出，程序结束了，但是并没有释放内存，这是为什么呢？？？

如下图：  
当我们执行useTrap函数时，注意，是没有结束此函数，boy和girl指针其实是被两个智能指针托管的，所以他们的引用计数是2  
![](https://i-blog.csdnimg.cn/blog_migrate/298ccf727050388799429c2056613ba8.png#pic_center)

useTrap函数结束后，函数中定义的智能指针被清掉，boy和girl指针的引用计数减1，还剩下1，对象中的智能指针还是托管他们的，所以函数结束后没有将boy和gilr指针释放的原因就是于此。  
![](https://i-blog.csdnimg.cn/blog_migrate/030d8d5c44901ae3bba4176168c8acfe.png#pic_center)

**所以在使用shared_ptr智能指针时，要注意避免对象交叉使用智能指针的情况！** 否则会导致内存泄露！

当然，这也是有办法解决的，那就是使用**weak_ptr**弱指针。

针对上面的情况，还讲一下另一种情况。如果是单方获得管理对方的共享指针，那么这样着是可以正常释放掉的！  
例如：

    void useTrap() {
    	shared_ptr<Boy> spBoy(new Boy());
    	shared_ptr<Girl> spGirl(new Girl());
    
    	// 单方获得管理
    	//spBoy->setGirlFriend(spGirl);
    	spGirl->setBoyFriend(spBoy);	
    }
    

![](https://i-blog.csdnimg.cn/blog_migrate/e948458859083e50c54f9a376f69e330.jpeg#pic_center)  
反过来也是一样的！

这是什么原理呢？

1.  首先释放spBoy，但是因为girl对象里面的智能指针还托管着boy，boy的引用计数为2，所以释放spBoy时，引用计数减1，boy的引用计数为1；
2.  在释放spGirl，girl的引用计数减1，为零，开始释放girl的内存，因为girl里面还包含有托管boy的智能指针对象，所以也会进行boyFriend的内存释放，boy的引用计数减1，为零，接着开始释放boy的内存。最终所有的内存都释放了。

* * *

五、weak_ptr
-----------

weak_ptr 设计的目的是为配合 shared_ptr 而引入的一种智能指针来协助 shared_ptr 工作, **它只可以从一个 shared_ptr 或另一个 weak_ptr 对象构造,** 它的构造和析构不会引起引用记数的增加或减少。 同时weak_ptr 没有重载\*和->但可以使用 **lock** 获得一个可用的 shared_ptr 对象。

1.  弱指针的使用；  
    weak_ptr wpGirl_1; // 定义空的弱指针  
    weak_ptr wpGirl_2(spGirl); // 使用共享指针构造  
    wpGirl_1 = spGirl; // 允许共享指针赋值给弱指针
    
2.  弱指针也可以获得引用计数；  
    wpGirl_1.use_count()
    
3.  弱指针不支持 \* 和 -> 对指针的访问；  
    ![](https://i-blog.csdnimg.cn/blog_migrate/b418dc4823b94013f0bdf03e5482a9b3.jpeg#pic_center)
    
4.  在必要的使用可以转换成共享指针 lock()；
    
        shared_ptr<Girl> sp_girl;
        sp_girl = wpGirl_1.lock();
        
        // 使用完之后，再将共享指针置NULL即可
        sp_girl = NULL;
        
    

使用代码：

    shared_ptr<Boy> spBoy(new Boy());
    shared_ptr<Girl> spGirl(new Girl());
    
    // 弱指针的使用
    weak_ptr<Girl> wpGirl_1;			// 定义空的弱指针
    weak_ptr<Girl> wpGirl_2(spGirl);	// 使用共享指针构造
    wpGirl_1 = spGirl;					// 允许共享指针赋值给弱指针
    
    cout << "spGirl \t use_count = " << spGirl.use_count() << endl;
    cout << "wpGirl_1 \t use_count = " << wpGirl_1.use_count() << endl;
    
    	
    // 弱指针不支持 * 和 -> 对指针的访问
    /*wpGirl_1->setBoyFriend(spBoy);
    (*wpGirl_1).setBoyFriend(spBoy);*/
    
    // 在必要的使用可以转换成共享指针
    shared_ptr<Girl> sp_girl;
    sp_girl = wpGirl_1.lock();
    
    cout << sp_girl.use_count() << endl;
    // 使用完之后，再将共享指针置NULL即可
    sp_girl = NULL;
    

当然这只是一些使用上的小例子，具体用法如下：

**请看Boy类**

    #include <iostream>
    #include <string>
    #include <memory>
    
    using namespace std;
    
    class Girl;
    
    class Boy {
    public:
    	Boy() {
    		cout << "Boy 构造函数" << endl;
    	}
    
    	~Boy() {
    		cout << "~Boy 析构函数" << endl;
    	}
    
    	void setGirlFriend(shared_ptr<Girl> _girlFriend) {
    		this->girlFriend = _girlFriend;
    
    
    		// 在必要的使用可以转换成共享指针
    		shared_ptr<Girl> sp_girl;
    		sp_girl = this->girlFriend.lock();
    
    		cout << sp_girl.use_count() << endl;
    		// 使用完之后，再将共享指针置NULL即可（局部申请的可自动释放）
    		sp_girl = NULL;
    	}
    
    private:
    	weak_ptr<Girl> girlFriend;
    };
    
    class Girl {
    public:
    	Girl() {
    		cout << "Girl 构造函数" << endl;
    	}
    
    	~Girl() {
    		cout << "~Girl 析构函数" << endl;
    	}
    
    	void setBoyFriend(shared_ptr<Boy> _boyFriend) {
    		this->boyFriend = _boyFriend;
    	}
    
    private:
    	shared_ptr<Boy> boyFriend;
    };
    
    
    void useTrap() {
    	shared_ptr<Boy> spBoy(new Boy());
    	shared_ptr<Girl> spGirl(new Girl());
    
    	spBoy->setGirlFriend(spGirl);
    	spGirl->setBoyFriend(spBoy);
    }
    
    
    int main(void) {
    	useTrap();
    
    	system("pause");
    	return 0;
    }
    

![](https://i-blog.csdnimg.cn/blog_migrate/5228085f009afcc71a2436dd7b90447c.jpeg#pic_center)

在类中使用弱指针接管共享指针，在需要使用时就转换成共享指针去使用即可！

自此问题完美解决！

### expired函数的用法

应评论区某位朋友的要求，现在加上weak_ptr指针的expired函数的用法!

expired：判断当前weak_ptr智能指针是否还有托管的对象，有则返回false，无则返回true

如果返回true，等价于 use_count() == 0，即已经没有托管的对象了；当然，可能还有析构函数进行释放内存，但此对象的析构已经临近（或可能已发生）。

**示例**  
演示如何用 expired 检查指针的合法性。  
在网上找了一段代码，加上自己的注释理解

    #include <iostream>
    #include <memory>
    
    std::weak_ptr<int> gw;
    
    void f() {
    
    	// expired：判断当前智能指针是否还有托管的对象，有则返回false，无则返回true
    	if (!gw.expired()) {
    		std::cout << "gw is valid\n";	// 有效的，还有托管的指针
    	} else {
    		std::cout << "gw is expired\n";	// 过期的，没有托管的指针
    	}
    }
    
    int main() {
    	{
    		auto sp = std::make_shared<int>(42);
    		gw = sp;
    
    		f();
    	}
    
    	// 当{ }体中的指针生命周期结束后，再来判断其是否还有托管的指针
    	f();
    
    	return 0;
    }
    

![](https://i-blog.csdnimg.cn/blog_migrate/6e41fc1c6e367559e356ff3850310f37.png#pic_center)

在 { } 中，sp的生命周期还在，gw还在托管着make_shared赋值的指针(sp)，所以调用f()函数时打印"gw is valid\\n";  
当执行完 { } 后，sp的生命周期已经结束，已经调用析构函数释放make_shared指针内存(sp)，gw已经没有在托管任何指针了，调用expired()函数返回true，所以打印"gw is expired\\n";

* * *

### weak_ptr常用情况
*   **观察者模式**：
    
    *   在观察者模式中，观察者需要引用被观察的对象，但不希望影响被观察对象的生命周期。使用 `std::weak_ptr` 可以确保被观察对象的生命周期由其他部分管理，而观察者只是被动地观察。
        
*   **缓存或间接引用**：
    
    *   当你需要缓存某个对象的引用，但不希望缓存本身影响对象的生命周期时，`std::weak_ptr` 是一个合适的选择。
        
*   **父子关系或复杂对象图**：
    
    *   在父对象和子对象的关系中，子对象可能需要引用父对象，但父对象的生命周期应该由其他部分管理。使用 `std::weak_ptr` 可以避免子对象影响父对象的生命周期。
 



六、智能指针的使用陷阱
-----------

1.  不要把一个原生指针给多个智能指针管理;
    
    int \*x = new int(10);  
    unique_ptr< int > up1(x);  
    unique_ptr< int > up2(x);  
    // 警告! 以上代码使up1 up2指向同一个内存,非常危险  
    或以下形式：  
    up1.reset(x);  
    up2.reset(x);
    
2.  记得使用u.release()的返回值;  
    在调用u.release()时是不会释放u所指的内存的，这时返回值就是对这块内存的唯一索引，如果没有使用这个返回值释放内存或是保存起来，这块内存就泄漏了.
    
3.  禁止delete 智能指针get 函数返回的指针;  
    如果我们主动释放掉get 函数获得的指针，那么智能 指针内部的指针就变成野指针了，析构时造成重复释放，带来严重后果!
    
4.  禁止用任何类型智能指针get 函数返回的指针去初始化另外一个智能指针！  
    shared_ptr< int > sp1(new int(10));  
    // 一个典型的错误用法 shared_ptr< int > sp4(sp1.get());
    

