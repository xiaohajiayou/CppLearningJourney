

1\. 简介
------

**gtest是什么？**： [gtest](https://so.csdn.net/so/search?q=gtest&spm=1001.2101.3001.7020)是google下面的一块跨平台测试框架，它是为C++测试而生成的，支持自动测试以及丰富的用户断言  
如果是希望学习python的测试框架，可以进入另一篇文章【[熟练掌握pytest 单元测试框架](https://blog.csdn.net/weixin_42125125/article/details/144057610)】中进行学习

2\. 安装
------

从源码安装
``` shell
    git clone git@github.com:google/googletest.git
    cd googletest
    mkdir build
    cmake ..
    make -j8
    sudo cmake install
  
```

则gtest安装到了通过`usr/local/lib`指定的位置

3\. 快速开始
--------

gtest通过`TEST()`来收集测试套，比如：
```c++
    TEST(TestSuiteName, TestName) {
      ... test body ...
    }
```  

则是定义了一个TestSuiteName.TestName的测试case，在测试体中，我们可以使用断言判断测试的正确性。gtest提供了很多的断言判断，`ASSERT_*`会在测试失败时，断言并终止当前应用，还有`EXPECT_*`则是值时生成失败的测试集，应用还会继续执行，并最终生成测试报告  
下面使用`ASSERT_EQ` 和`EXPECT_EQ`举一个简单的例子
```c++
    ASSERT_EQ(x.size(), y.size()) << "Vectors x and y are of unequal length";
    
    for (int i = 0; i < x.size(); ++i) {
      EXPECT_EQ(x[i], y[i]) << "Vectors x and y differ at index " << i;
    }
```    

而真正的测试可能如下所示：
```c++
    // Tests factorial of 0.
    TEST(FactorialTest, HandlesZeroInput) {
      EXPECT_EQ(Factorial(0), 1);
    }
    
    // Tests factorial of positive numbers.
    TEST(FactorialTest, HandlesPositiveInput) {
      EXPECT_EQ(Factorial(1), 1);
      EXPECT_EQ(Factorial(2), 2);
      EXPECT_EQ(Factorial(3), 6);
      EXPECT_EQ(Factorial(8), 40320);
    }
```    

如果测试套`TestSuiteName`名字一样，则会被归纳成一类测试集，后面可以使用`--gtest_filter=FactorialTest.*`进行自动测试

4\. Fixtures
------------

有时候我们希望做一些测试前`SetUp`的准备工作(比如连上服务器)，以及测试后`TearDown`的一些资源回收，此时我们可以用到gtest的Fixtures能力  
要去创建一个Fixtures，需要以下几个步骤:

1.  需要创建一个类继承testing::Test. 在此类的测试体中，就会继承很多`protect`的fixture 函数
2.  在此类测试体中，我们就可以声明各种测试代码
3.  如果有必要，可以在`默认构造函数`或者`SetUp`函数中来做一些数据准备工作
4.  如果有必要，在`析构函数`或者`TearDown`中来做一些资源回收的收尾工作  
    如果使用fixture, 则需要使用`TEST_F`而不是`TEST`来测试代码，例如：
```c++
    class FixtureTest: public testing::Test {
     protected:
      FixtureTest() {
       	array = new int[1024];
      }
    
    void set(const int index, const int value) { array[index] = value; }
     int get(const int index) const { return array[index]; }
    
      ~FixtureTest() override {
      delete[] array;
    }
    
      int *array;
    };
    
    TEST_F(FixtureTest, SetAndGet) {
      ASSERT_NE(array, nullptr);
      set(1, 10);
      EXPECT_EQ(get(1), 10);
    }
    

// 运行结果：

//     [==========] Running 1 test from 1 test suite.
//     [----------] Global test environment set-up.
//     [----------] 1 test from FixtureTest
//     [ RUN      ] FixtureTest.SetAndGet
//     [       OK ] FixtureTest.SetAndGet (0 ms)
//     [----------] 1 test from FixtureTest (0 ms total)
    
//     [----------] Global test environment tear-down
//     [==========] 1 test from 1 test suite ran. (0 ms total)
//     [  PASSED  ] 1 tes
    
```
5\. 在`main`函数中调用
----------------

上面讲到的`TEST()`和`TEST_F()`都是隐式地注册测试集到gtest中，并且不需要main函数就可以自动运行，其原因是在库`libgetst_main.a`中隐藏了一个main函数，如果我们想自己控制运行，则只需要去掉`gtest_main`，并且在自己的main函数中调用RUN_ALL_TESTS()即可，代码如下：
```c++
    int main(int argc, char **argv) {
      testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
    }
```    

这里的`testing::InitGoogleTest()`是为了解析可执行的传参，比如`--gtest_filter`，而`RUN_ALL_TESTS()`则是运行所有测试集

6\. gtest测试参数
-------------

gtest支持的参数非常多，建议使用`--gtest_help`进行查看，下面重点介绍几个我常用到的参数

*   –gtest_list_tests: 列出所有测试集，但不会运行
*   –gtest_filter=POSITIVE_PATTERNS[-NEGATIVE_PATTERNS]: 过滤测试集，在`-`之前表示要运行哪些测试，`-`之后表示当前运行的测试中过滤所有`NEGATIVE_PATTERNS`的测试
*   –gtest_repeat=[COUNT]：指定测试重复运行`COUNT`次
*   –gtest_output=(json|xml)[:DIRECTORY_PATH/|:FILE_PATH]： 测试结果报告输出成汇报文档，支持json和xml两种格式