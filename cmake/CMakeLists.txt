# CMake首先需要设置依赖包的最早版本 
cmake_minimum_required(3.3.10)

# 设置项目句柄，类似于IDE新建项目需要项目名
project(NAME)

# 原材料地址
add_subdirectory(./src)

# 减少头文件重复信息，将路径放在cmake管理(该方法出现在顶层cmake，会导致所有源文件的编译包含多余头文件)
include_directories(./include)

# 菜名 源文件地址
add_executable(main ./main.cpp)

#编译静态库和动态库
add_library(static_lib ./static_lib.cpp)
add_library(shared_lib SHARED ./shared_lib.cpp)

# 链接

# 链接静态库的方式,两种方法等价
link_directories(./lib)
target_link_libraries(main static_lib)

link_libraries(./lib/static_lib)


# 需要注意的是，在windows中使用visualstudio编译器，编译动态库时，会生成shared.lib和shared.dll
# cmake中需要链接.lib，但是需要将.dll放到可执行文件夹活环境变量中。
 