
namespace factory {
class AnimalFactory {
 public:
    enum AnimalType {
        DOG,
        CAT
    }
    ~AnimalFactory();
    AnimalBase* createAnimal();
 private:
};

class AnimalBase {
 public:
    virtual ~AnimalBase();
    virtual void eat();
 private:
};


class Dog : public AnimalBase {
 public:
    Dog();
    ~Dog() override;
    void eat() override;
 private:
};
}  // namespace factory
