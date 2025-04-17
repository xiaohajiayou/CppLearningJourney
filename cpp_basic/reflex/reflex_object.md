## 原理
- 核心类为classinfo，这个类通过宏，声明为静态成员变量sclass_info（需要在类外初始化，且由于是运行时检查，需传参的构造函数需要用强制转换，将派生类对象转换成需要的基类对象，使其对象使用派生类的虚函数表）。
- 类外初始化sclass_info时，传参会自动调用classinfo类的注册函数，本质上为调用ReflexObjectEx<T>::Register(*this);
- 在ReflexObjectEx<T>::Register(*this)中，考虑到classinfo的模版参数可能已经是反射类（即新建类为反射派生类类的子类），故需要做一次static_cast转为基类，然后调用ReflexObject::Register，将class_info类注册到静态的哈希表中
- 当需要创建实例时，反射类和反射类的派生类，分别用ReflexObjectEx<MyDerivedClass>::CreateObject("MyDerivedClass")直接从哈希表中拿出对应的class_info对象，然后调用其中的CreateObject方法，创建统一的基类对象，然后调用dynamic_cast类型转换为需要的派生类指针类型。

## 注意事项
- 反射基本是通过静态成员初始化机制和模版、多态来实现的
- 直接在反射机制中全部使用父类指针，适合基类高度抽象，派生类基本只需要调用通用的虚函数
- 如果在业务场景中，各模块需要使用的接口和功能差异较大，则需要使用模版元编程，最后将基类指针转回派生类指针

## 头文件

 #ifndef MODULES_INFERENCE_INCLUDE_REFLEX_OBJECT_H_
 #define MODULES_INFERENCE_INCLUDE_REFLEX_OBJECT_H_
 
 #include <functional>
 #include <map>
 #include <memory>
 #include <string>
 
 #define DECLARE_REFLEX_OBJECT(Class)                              \
  public:                                                          \
   static cnstream::ClassInfo<cnstream::ReflexObject> sclass_info; \
                                                                   \
  protected:                                                       \
   const cnstream::ClassInfo<cnstream::ReflexObject>& class_info() const;
 
 #define IMPLEMENT_REFLEX_OBJECT(Class)                                                                                \
   cnstream::ClassInfo<ReflexObject> Class::sclass_info(std::string(#Class),                                           \
                                                        cnstream::ObjectConstructor<cnstream::ReflexObject>([]() {     \
                                                          return reinterpret_cast<cnstream::ReflexObject*>(new Class); \
                                                        }),                                                            \
                                                        true);                                                         \
   const cnstream::ClassInfo<cnstream::ReflexObject>& Class::class_info() const { return sclass_info; }
 
 #define DECLARE_REFLEX_OBJECT_EX(Class, BaseType)   \
  public:                                            \
   static cnstream::ClassInfo<BaseType> sclass_info; \
                                                     \
  protected:                                         \
   const cnstream::ClassInfo<BaseType>& class_info() const;
 
 #define IMPLEMENT_REFLEX_OBJECT_EX(Class, BaseType)                                                          \
   cnstream::ClassInfo<BaseType> Class::sclass_info(                                                          \
       std::string(#Class),                                                                                   \
       cnstream::ObjectConstructor<BaseType>([]() { return reinterpret_cast<BaseType*>(new Class); }), true); \
   const cnstream::ClassInfo<BaseType>& Class::class_info() const { return sclass_info; }
 
 namespace cnstream {
 
 /*****************************************
  * [T]: The return type for reflection object.
  *****************************************/
 
 template <typename T>
 using ObjectConstructor = std::function<T*()>;
 
 template <typename T>
 class ClassInfo {
  public:
   ClassInfo(const std::string& name, const ObjectConstructor<T>& constructor, bool register = false);
 
   T* CreateObject() const;
 
   std::string name() const;
 
   bool Register() const;
 
   const ObjectConstructor<T>& constructor() const;
 
  private:
   std::string name_;
   ObjectConstructor<T> constructor_;
 };  // class classinfo
 
 class ReflexObject {
  public:
   static ReflexObject* CreateObject(const std::string& name);
 
   static bool Register(const ClassInfo<ReflexObject>& info);
 
   virtual ~ReflexObject() = 0;
 #ifdef UNIT_TEST
   static void Remove(const std::string& name);
 #endif
 };  // class reflexobject<void>
 
 template <typename T>
 class ReflexObjectEx : public ReflexObject {
  public:
   static T* CreateObject(const std::string& name);
 
   static bool Register(const ClassInfo<T>& info);
 
   virtual ~ReflexObjectEx() = 0;
 };  // class reflectobject
 
 template <typename T>
 ClassInfo<T>::ClassInfo(const std::string& name, const ObjectConstructor<T>& constructor, bool call_register)
     : name_(name), constructor_(constructor) {
   if (call_register) {
     Register();
   }
 }
 
 template <typename T>
 inline std::string ClassInfo<T>::name() const {
   return name_;
 }
 
 template <typename T>
 inline const ObjectConstructor<T>& ClassInfo<T>::constructor() const {
   return constructor_;
 }
 
 template <typename T>
 inline bool ClassInfo<T>::Register() const {
   return ReflexObjectEx<T>::Register(*this);
 }
 
 template <typename T>
 T* ClassInfo<T>::CreateObject() const {
   if (NULL != constructor_) {
     return constructor_();
   }
   return nullptr;
 }
 
 template <typename T>
 T* ReflexObjectEx<T>::CreateObject(const std::string& name) {
   auto ptr = ReflexObject::CreateObject(name);
   if (nullptr == ptr) return nullptr;
   T* ret = dynamic_cast<T*>(ptr);
   if (nullptr == ret) {
     delete ptr;
     return nullptr;
   }
   return ret;
 }
 
 template <typename T>
 bool ReflexObjectEx<T>::Register(const ClassInfo<T>& info) {
   // build base ClassInfo(ClassInfo<ReflexObjectEx>)
   ObjectConstructor<ReflexObject> base_constructor = NULL;
   if (info.constructor() != NULL) {
     base_constructor = [info]() { return static_cast<ReflexObject*>(info.constructor()()); };
   }
   ClassInfo<ReflexObject> base_info(info.name(), base_constructor);
 
   return ReflexObject::Register(base_info);
 }
 
 template <typename T>
 ReflexObjectEx<T>::~ReflexObjectEx() {}
 
 }  // namespace cnstream
 
 #endif  // MODULES_INFERENCE_INCLUDE_REFLEX_OBJECT_HPP_
 

 ## 源代码
```c++
/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#include "reflex_object.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "cnstream_logging.hpp"

namespace cnstream {

static std::map<std::string, ClassInfo<ReflexObject>> sg_obj_map;

ReflexObject* ReflexObject::CreateObject(const std::string& name) {
  const auto& obj_map = sg_obj_map;
  auto info_iter = obj_map.find(name);

  if (obj_map.end() == info_iter) return nullptr;

  const auto& info = info_iter->second;
  return info.CreateObject();
}

bool ReflexObject::Register(const ClassInfo<ReflexObject>& info) {
  auto& obj_map = sg_obj_map;
  if (obj_map.find(info.name()) != obj_map.end()) {
    LOGI(REFLEX_OBJECT) << "Register object named [" << info.name() << "] failed!!!"
                        << "Object name has been registered.";
    return false;
  }

  obj_map.insert(std::pair<std::string, ClassInfo<ReflexObject>>(info.name(), info));

  LOGI(REFLEX_OBJECT) << "Register object named [" << info.name() << "]";
  return true;
}

ReflexObject::~ReflexObject() {}

#ifdef UNIT_TEST
void ReflexObject::Remove(const std::string& name) {
  auto& obj_map = sg_obj_map;
  auto info_iter = obj_map.find(name);

  if (obj_map.end() != info_iter) {
    obj_map.erase(name);
  }
}
#endif

}  // namespace cnstream

```
