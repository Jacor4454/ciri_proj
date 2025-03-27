#include <gtest/gtest.h>

#include "../../src/class_layer/perceptron/class_perceptron.h"

namespace my {
namespace project {
namespace {

// The fixture for testing class Foo.
class PerceptronLayerTest : public testing::Test {
    protected:
    // You can remove any or all of the following functions if their bodies would
    // be empty.

    PerceptronLayerTest() {
        // You can do set-up work for each test here.
        tensor::threads_setWorkers(4);
        tensor::threads_initaliseThreads();
    }

    ~PerceptronLayerTest() override {
        // You can do clean-up work that doesn't throw exceptions here.
        tensor::threads_killThreads();
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp() override {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Class members declared here can be used by all tests in the test suite
};

// Tests that the Foo::Bar() method does Abc.
TEST_F(PerceptronLayerTest, MakeLayer) {
    BaseLayer* b = new perceptron({1, 784}, {1, 150});
    EXPECT_EQ(b->getLayerType(), "Perceptron");
    delete b;
}
TEST_F(PerceptronLayerTest, forward) {
    int n = 1;
    int m = 150;
    int k = 784;
    BaseLayer* b = new perceptron({n,k}, {n,m});
    tensor input({n,k});
    tensor output({n,m});
    b->forward(output, input);
    delete b;
}
TEST_F(PerceptronLayerTest, backwards) {
    int n = 1;
    int m = 150;
    int k = 784;
    BaseLayer* b = new perceptron({n,k}, {n,m});
    tensor dinput({n,k});
    tensor input({n,k});
    tensor doutput({n,m});
    tensor output({n,m});
    tensor dnm({});
    b->backward(dinput, input, doutput, dnm, output);
    delete b;
}
TEST_F(PerceptronLayerTest, learn) {
    int n = 1;
    int m = 150;
    int k = 784;
    BaseLayer* b = new perceptron({n,k}, {n,m});
    b->learn(0.4);
    delete b;
}



}
}
}

