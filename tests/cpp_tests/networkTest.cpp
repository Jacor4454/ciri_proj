#include <gtest/gtest.h>

#include "../../src/class_network/class_learning_network.h"

namespace my {
namespace project {
namespace {

// The fixture for testing class Foo.
class NetworkTest : public testing::Test {
    protected:
    // You can remove any or all of the following functions if their bodies would
    // be empty.

    NetworkTest() {
        // You can do set-up work for each test here.
        tensor::threads_setWorkers(4);
        tensor::threads_initaliseThreads();
    }

    ~NetworkTest() override {
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

TEST_F(NetworkTest, MakeNetwork) {
    network myNetwork(inputDefObject({1,784}), {layerDefObject({1,150}, layers::recursive, activations::ReLU)}, outputDefObject({1,10}, layers::perceptron, activations::Sigmoid));
    tensor myTensor1({1, 784});
    myNetwork.forward({myTensor1});
}

TEST_F(NetworkTest, MakeLearningNetwork) {
    learningNetwork myNetwork(inputDefObject({1,784}), {layerDefObject({1,150}, layers::recursive, activations::ReLU)}, outputDefObject({1,10}, layers::perceptron, activations::Sigmoid), 2);
    tensor myTensor1a({1, 784});
    tensor myTensor1b({1, 784});
    myNetwork.forward({myTensor1a, myTensor1b});
    tensor myTensor2a({1, 10});
    tensor myTensor2b({1, 10});
    myNetwork.backward({myTensor2a, myTensor2b});
}


}
}
}




