#include <gtest/gtest.h>

#include "../../src/class_layer/recersive/class_recersive.h"

namespace my {
namespace project {
namespace {

// The fixture for testing class Foo.
class RecursiveLayerTest : public testing::Test {
    protected:
    // You can remove any or all of the following functions if their bodies would
    // be empty.

    RecursiveLayerTest() {
        // You can do set-up work for each test here.
        tensor::threads_setWorkers(4);
        tensor::threads_initaliseThreads();
    }

    ~RecursiveLayerTest() override {
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
TEST_F(RecursiveLayerTest, MakeLayer) {
    BaseLayer* b = new recursive({1,784}, {1,150});
    EXPECT_EQ(b->getLayerType(), "Recersive");
    delete b;
}
TEST_F(RecursiveLayerTest, forward) {
    int n = 1;
    int m = 150;
    int k = 784;
    BaseLayer* b = new recursive({n,k}, {n,m});
    tensor input({n,k});
    tensor output({n,m});
    b->forward(output, input);
    b->forward(output, input);
    delete b;
}
TEST_F(RecursiveLayerTest, backwards) {
    int n = 1;
    int m = 150;
    int k = 784;
    BaseLayer* b = new recursive({n,k}, {n,m});
    tensor dinput({n,k});
    tensor input({n,k});
    tensor doutput({n,m});
    tensor output({n,m});
    tensor previous_output({n,m});
    b->backward(dinput, input, doutput, previous_output, output);
    delete b;
}
TEST_F(RecursiveLayerTest, learn) {
    int n = 1;
    int m = 150;
    int k = 784;
    BaseLayer* b = new recursive({n,k}, {n,m});
    b->learn();
    delete b;
}



}
}
}

