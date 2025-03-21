#include <iostream>
#include <string>
#include <memory>
using namespace std;

void MemoryLeak1() {
    String *str = new string("申请堆上内存");
    // 没有delete
}

void MemoryLeak2() {
    String *str = new string("申请堆上内存");
    int error = DoSomething();
    if(error) {
        return -1;
        // 意外退出，没有delete
    }

    delete str;
}

class Resource {
public:
    Resource(int value) : m_data(value) {
        cout << "申请资源" << endl;
    }

    ~Resource() {
        cout << "释放资源" << endl;
    }

private:
    int m_data;
};



int main() {
    // 1、内存泄漏示范
    MemoryLeak1();
    MemoryLeak2();

    // 2、使用智能指针
    auto_pre<Resource> p1(new Resource(1));
    // p1赋值给p2后，p2原先指针被释放掉，接受托管p1的指针，然后p1被置为空指针
    auto_pre<Resource> p2 = p1;


    // 3、由于STL容器中元素需要支持可复制和可赋值，所以auto_ptr存在风险
    vector<auto_ptr<int>> vec;
    auto_ptr<int> p3(new int(1));
    auto_ptr<int> p4(new int(2));
    // 注意必须修饰为右值，才能插入容器
    vec.push_back(move(p3));
    vec.push_back(move(p4));
    // 风险来了，元素指向了野指针
    vec[0] = vec[1];


    // 4、unique_ptr在STL容易中修复了auto_ptr的问题
    vector<unique_ptr<int>> vec2;
    unique_ptr<int> p5(new int(1));
    unique_ptr<int> p6(new int(2));
    // 插入容器同样需要右值
    vec2.push_back(move(p5));
    vec2.push_back(move(p6));
    // 容器内赋值也需要move，使用户注意后果不会出现auto_ptr的问题
    vec2[0] = std::move(vec2[1]);
    // unique_ptr支持对象数组的内存管理，自动调用delete释放
    unique_ptr<int[]> p7(new int[10]);


    // 5、auto_ptr和unique_ptr这种排他型内存管理，无法满足变量共享情况
    shared_ptr<Resource> sp1;
    shared_ptr<Resource> sp2(new Resource(1));

    cout<<"sp1的引用计数："<<sp1.use_count()<<endl;
    cout<<"sp2的引用计数："<<sp2.use_count()<<endl;
    // 0 1

    sp1 = sp2;
    cout<<"sp1的引用计数："<<sp1.use_count()<<endl;
    cout<<"sp2的引用计数："<<sp2.use_count()<<endl;
    // 2 2

    shared_ptr<Resource> sp3(sp2);
    cout<<"sp1的引用计数："<<sp1.use_count()<<endl;
    cout<<"sp2的引用计数："<<sp2.use_count()<<endl;
    cout<<"sp3的引用计数："<<sp3.use_count()<<endl;
    // 3 3 3

    // 6、申请shared_ptr建议使用make_shared接口，其会申请一块连续的内存，分别管理对象部分和控制块部分
    // 该方式减少一次内存分配。且由于内存连续，容易命中缓存，性能更好
    shared_ptr<Resource> sp4 = make_shared<Resource>(1);

    return 0;
}

// 内存泄漏的例子

