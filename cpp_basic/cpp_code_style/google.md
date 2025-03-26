* * *

  
![](https://i-blog.csdnimg.cn/blog_migrate/5c8468ac1ab52344b09dc3de99066c72.png)

* * *

本文参考 [Google开源项目风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/) ，由于原文篇幅过长，本文对其进行精简，读者可以通过右侧目录进行导航阅读。  
本文对一些重点进行了红色标注 ，同时为了便于理解，还进行了大量举例。

* * *

**一.头文件**
---------

通常每一个 `.cc` 文件都有一个对应的 `.h` 文件. 也有一些常见例外, 如[单元测试](https://so.csdn.net/so/search?q=%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95&spm=1001.2101.3001.7020)代码和只包含 `main()` 函数的 `.cc` 文件.  

### #define 保护

所有[头文件](https://so.csdn.net/so/search?q=%E5%A4%B4%E6%96%87%E4%BB%B6&spm=1001.2101.3001.7020)都应该使用 `#define` 来防止头文件被多重包含, 命名格式当是: `<PROJECT>_``<PATH>``_``<FILE>``_H_`  
例如：  
#ifndef FOO\_BAR\_BAZ\_H\_  
#define FOO\_BAR\_BAZ\_H\_  
…  
#endif _// FOO\_BAR\_BAZ\_H_\_  

### 前置声明

所谓「前置声明」（forward declaration）是类、函数和模板的纯粹声明，没伴随着其定义.

      - 尽量避免前置声明那些定义在其他项目中的实体.
      - 函数：总是使用 `#include`.
      - 类模板：优先使用 `#include`.
    

### 内联函数

当函数被声明为内联函数之后, 编译器会将其内联展开, 而不是按通常的函数调用机制进行调用.  
只有当函数只有 10 行甚至更少时才将其定义为内联函数.  

### **包含文件的名称及次序**

将包含次序标准化可增强可读性、避免隐藏依赖（hidden dependencies，注：隐藏依赖主要是指包含的文件编译），次序如下：

         1. 当前cpp文件对应的.h文件
         1. C 系统文件
         1. C++ 系统文件
         1. 其他库的 .h 文件
         1. 本项目内 .h 文件
    

  
举例来说，google-awesome\-project/src/foo/internal/fooserver.cc 的包含次序如下：  
#include “foo/public/fooserver.h” _// 优先位置　　 _#include <sys/types.h> _//C系统文件_  
　　 #include <unistd.h> /_/C系统文件_

#include <hash\_map> /\_/C++系统文件 \_  
　　 #include \_//C++系统文件 \_

#include “base/basictypes.h” //其他库.h文件  
　　 #include “base/commandlineflags.h” //其他库.h文件  
　　 #include “foo/public/bar.h” /_/本项目的.h文件_\*\*  

二.作用域
-----

### 命名空间

命名空间将全局作用域细分为独立的, 具名的作用域, 可有效防止全局作用域的命名冲突.

         - 遵守 命名空间命名 中的规则
         - 在命名空间的最后注释出命名空间的名字
    

例如：  
_// .h 文件_  
namespace mynamespace {  
  
  
_// 所有声明都置于命名空间中_  
_// 注意不要使用缩进_  
class MyClass {  
public:  
…  
void Foo();  
};  
  
  
} _// namespace mynamespace_

         - 不应该使用 _using 指示_ 引入整个命名空间的标识符号,例如： ~~using namespace~~~~ ~~~~std~~~~；~~
         - 不要在头文件中使用 命名空间别名  例如：~~ namespace baz = ::foo::bar::baz;~~
         - 禁止用内联命名空间  例如： ~~inline namespace foo{...}~~
    

三.类
---

### 构造函数的职责

构造函数中不允许调用虚函数，如果代码允许，应在构造函数出错直接终止程序，否则使用Init函数进行初始化  

### 删除默认拷贝构造和赋值运算符

如果定义的类型明确不允许使用拷贝和赋值操作，需要使用delete关键字禁用  
例如 ：  
MyClass(**const** MyClass&) = **delete**; //禁用拷贝构造  
MyClass& **operator**\=(**const** MyClass&) = **delete**; //禁用赋值运算符  

### 结构体和类的使用时机

仅当只有数据成员时使用 `struct`, 其它一概使用 `class`.  

### 继承

      - 所有继承必须是 `public` 的. 如果你想使用私有继承, 你应该替换成把基类的实例作为成员对象的方式
      - 析构函数应声明为 virtual 
      - 对于可能被子类访问的成员函数, 不要过度使用 `protected` 关键字. 注意, 数据成员都必须是 [私有的](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/classes/#access-control).
      - 对于重载的虚函数或虚析构函数, 使用 `override`, 或 (较不常用的) `final` 关键字显式地进行标记
      - 不允许多继承，如果需要多继承，除第一个类外，其他都应该是接口类型
    

### 运算符重载

除少数特定环境外, 不要重载运算符. 也不要创建用户定义字面量.  

四.函数\*\*
--------

      - 将所有输入参数（不修改值的参数）放在输出参数（会修改值并返回给调用方的参数）之前，常数参数要使用 const 修饰 
      - 所有引用参数必须加上const
    

`例如 ： void Example(``const int arg1`` ，``const``int& arg2``，void* arg3``);`

      - 如果需要重载函数，应写在临近的代码位置
      - 缺省参数应该置于非缺省函数之后
    

五.命名约定
------

### 通用命名规则

函数命名, 变量命名, 文件命名要有描述性; 少用缩写.  

### 文件命名

文件名要全部小写, 可以包含下划线 (`_`) 或连字符 (`-`), 依照项目的约定. 如果没有约定, 那么 “`_`” 更好.  
例如 ：

      - `my_useful_class.cc`
      - `my-useful-class.cc`
    

### 类型命名

类型名称的每个单词首字母均大写, 不包含下划线，如： `MyExcitingClass`, `MyExcitingEnum`  

### 变量命名

变量 (包括函数参数) 和数据成员名一律小写, 单词之间用下划线连接. 类的成员变量以下划线结尾, 但结构体的就不用,不使用匈牙利命名法，驼峰命名法  

#### 普通变量

如: `a_local_variable`, `a_struct_data_member`, `a_class_data_member_`  

#### 类成员变量

类成员变量最后应该接下划线\_  
如：  
**class** **TableInfo** {  
**private**:  
string table\_\_name\_\_ ; _// 好 - 后加下划线._  
string tablename\_ ; _// 好._  
**static** Pool\* pool\_ ; _// 好._  
};  

#### 结构体变量

结构体的成员变量可以和普通变量一样，, 不用像类那样接下划线:  
如：  
**struct** UrlTableProperties {  
string name;  
int num\_entries;  
**static** Pool\* pool;  
};  

#### 常量命名

常量命名需要以小写字母“k”开头，不使用下划线，单词首字母大写，  
如： **const** int kDaysInAWeek = 7;  

#### 函数命名

函数命名采用 驼峰命名规则，即单词首字母大写，不适用下划线  
如： AddTableEntry()  
DeleteUrl()  
OpenFileOrDie()  

#### 命名空间命名

命名空间以小写字母命名，并遵守通用命名规则最高层的命名空间的名字取决于项目名称。命名空间中的代码，应该存放在和命名空间的名字匹配的文件夹中  

#### 枚举命名

枚举的命名应和宏一致，全部使用大写字母，单词间使用下划线分隔  
如：**enum** AlternateUrlTableErrors {  
OK = 0,  
OUT\_OF\_MEMORY = 1,  
MALFORMED\_INPUT = 2,  
};  

#### 宏命名

宏命名全部使用大写字母，单词间使用下划线分隔，如 ：  
#define ROUND(x) …  
#define PI\_ROUNDED 3.0  

六.注释
----

### 注释风格

注释使用 // 或/\*\*/都可以，与团队保持一致  

### 文件注释

每个文件开头应加入版权公告，如果文件仅仅是一些测试代码，可以不使用文件注释  

### 法律公告和作者信息

每个文件都应该包含许可证引用. 为项目选择合适的许可证版本.(比如, Apache 2.0, BSD, LGPL, GPL)  

### 文件内容

如果一个 `.h` 文件声明了多个概念（类型）, 则文件注释应当对文件的内容做一个大致的说明, 同时说明各概念之间的联系. 一个一到两行的文件注释就足够了, 对于每个概念的详细文档应当放在各个概念中, 而不是文件注释中.  

### 类注释

每个类的定义都要附带一份注释, 描述类的功能和用法, 除非它的功能相当明显.  
如：  
_// Iterates over the contents of a GargantuanTable._  
_// Example:_  
_// GargantuanTableIterator\* iter = table->NewIterator();_  
_// for (iter->Seek(“foo”); !iter->done(); iter->Next()) {_  
_// process(iter->key(), iter->value());_  
_// }_  
_// delete iter;_  
**class** **GargantuanTableIterator** {  
…  
};  

### 函数注释

函数声明处的注释描述函数功能; 定义处的注释描述函数实现  

#### 函数声明

函数声明处注释的内容:

         - 函数的输入输出.
         - 对类成员函数而言: 函数调用期间对象是否需要保持引用参数, 是否会释放这些参数.
         - 函数是否分配了必须由调用者释放的空间.
         - 参数是否可以为空指针.
         - 是否存在函数使用上的性能隐患.
    

举例如下:  
_// Returns an iterator for this table. It is the client’s_  
_// responsibility to delete the iterator when it is done with it,_  
_// and it must not use the iterator once the GargantuanTable object_  
_// on which the iterator was created has been deleted._  
_//_  
_// The iterator is initially positioned at the beginning of the table._  
_//_  
_// This method is equivalent to:_  
_// Iterator\* iter = table->NewIterator();_  
_// iter->Seek("");_  
_// return iter;_  
_// If you are going to immediately seek to another place in the_  
_// returned iterator, it will be faster to use NewIterator()_  
_// and avoid the extra seek._  
Iterator\* GetIterator() **const**;  
对于简单的函数可简单注释  

#### 函数定义

如果函数的实现比较巧妙或复杂，应该注释清函数的实现思路  

### 变量注释

通常变量名本身足以很好说明变量用途. 某些情况下, 也需要额外的注释说明.  

七.C++的一些特性
----------

### 类型转换

不要使用强制类型转换或隐式类型转换，应该使用C++的类型转换 ，如 static\_cast  

### 自增和自减

对于迭代器和其他模板对象使用前置形式的自增或自减 ，因为前置自增效率相比后置自增效率更高。如 ++i ，–i，++it，–it 等  

### 预处理宏

      - 尽量不要在.h文件中定义宏
      - 尽量在马上要使用时才进行#define，使用后要立即#undef
      - 不要只是对已经存在的宏使用#undef，选择一个不会冲突的名称；
      - 不要试图使用展开后会导致 C++ 构造不稳定的宏, 不然也至少要附上文档说明其行为.
      - 尽量不要用##处理函数，类和变量的名字
    

### 空值，0 ， nullptr 和 NULL

      - 整数使用0，实数使用0.0
      - 指针用nullptr或者NULL
      - 字符串使用 '\0'
    

### sizeof的使用

尽量使用sizeof(varname)而不是sizeof(type)  

### auto的使用

      - auto只能在局部变量里使用，不要用在文件作用域
      - 如果auto 表达式的右值是一个对象，使用auto& ，例如 auto& person = Person（）；
      - 推荐在迭代遍历中使用 auto
    

### lambda表达式

      - 不要使用默认捕获，捕获要显示写出来 ,比起 `[=](int x) {return x + n;}`, 您该写成 `[n](int x) {return x + n;}` 才对
      - lambda表达式应该尽量简短，函数体不应过长（不超过10行）
      - lambda表达式的返回值应该以尾置返回类型的方式写出来，例如 [](int x, int y)->int{return x+y;};
    

### 与零值的判断

      - 整数与零值的判断应写为 if （num == 0）；
      - 浮点数与零值的判断应写为 if （num <0.00001&&num>-0.00001）; 不能写成 if（num == 0.0）；
      - 布尔值与零值判断应尽量写为 if（bool == false）；而不是if（!bool）；