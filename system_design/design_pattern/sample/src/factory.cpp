#include "factory.h"
#include<iostream>

namespace factory {

AnimalBase AnimalFactory::createAnimal(AnimalType type) {
    switch(type) {
        case DOG:  return new Dog;
        default: return nullptr;
    }
    
}

void Dog::eat() {
    std::cout<< "I am a dog" <<std::endl;
}

}  // namespace factory