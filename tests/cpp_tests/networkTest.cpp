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

        log::open("log.txt");
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
    Network myNetwork(inputDefObject({1,784}), {layerDefObject({1,150}, layers::recursive, activations::ReLU)}, outputDefObject({1,10}, layers::perceptron, activations::Sigmoid), 2);
    tensor myTensor1a({1, 784});
    tensor myTensor1b({1, 784});
    myNetwork.forward({myTensor1a, myTensor1b});
    tensor myTensor2a({1, 10});
    tensor myTensor2b({1, 10});
    myNetwork.backward({myTensor2a, myTensor2b});
}

TEST_F(NetworkTest, NetworkResize) {
    Network myNetwork(inputDefObject({1,784}), {layerDefObject({1,150}, layers::recursive, activations::ReLU)}, outputDefObject({1,10}, layers::perceptron, activations::Sigmoid), 1);
    tensor myTensor1a({1, 784});
    tensor myTensor1b({1, 784});
    myNetwork.forward({myTensor1a, myTensor1b}); // resizes from 1 to 2, should throw no errors
    tensor myTensor2a({1, 10});
    tensor myTensor2b({1, 10});
    EXPECT_THROW(myNetwork.backward({myTensor2a}), std::runtime_error); // cos forward was 2, this will throw
    myNetwork.backward({myTensor2a, myTensor2b}); // but this will pass
}

TEST_F(NetworkTest, NetworkLearn) {
    Network myNetwork(inputDefObject({1,784}), {layerDefObject({1,150}, layers::recursive, activations::ReLU)}, outputDefObject({1,10}, layers::perceptron, activations::Sigmoid), 1);
    tensor myTensor1a({1, 784});
    tensor myTensor1b({1, 784});
    tensor myTensor2a({1, 10});
    tensor myTensor2b({1, 10});
    myNetwork.learn({{myTensor1a, myTensor1b}}, {{myTensor2a, myTensor2b}});
}

TEST_F(NetworkTest, NetworkOutput) {
    Network myNetwork(inputDefObject({1,2}), {layerDefObject({1,20}, layers::recursive, activations::ReLU)}, outputDefObject({1,1}, layers::perceptron, activations::Sigmoid), 1);
    
    int k = 8;

    std::vector<std::vector<tensor>> input(1, std::vector<tensor>(8, tensor({1, 2})));
    std::vector<std::vector<tensor>> correct(1, std::vector<tensor>(8, tensor({1, 1})));

    int val = rand() & ((1<<(k-1))-1);
    int val2 = rand() & ((1<<(k-1))-1);
    int valc = val+val2;
    for(int j = 0; j < k; j++){
        input[0][j][0] = val & 1;
        input[0][j][1] = val2 & 1;
        correct[0][j][0] = valc & 1;

        val >>= 1;
        val2 >>= 1;
        valc >>= 1;
    }

    myNetwork.learn(input, correct);
}

#ifdef NETWORK_FULLLEARN
TEST_F(NetworkTest, NetworkFullLearn) {
    Network myNetwork(inputDefObject({1,2}), {layerDefObject({1,10}, layers::recursive, activations::Sigmoid, new AdamLearnerSelector(0.01,0.9,0.999))}, outputDefObject({1,1}, layers::perceptron, activations::Sigmoid, errors::MSE, new AdamLearnerSelector(0.01,0.9,0.999)), 1, 1231);
    
    int n = 50000;
    int k = 8;

    std::vector<std::vector<tensor>> input(n, std::vector<tensor>(k, tensor({1, 2})));
    std::vector<std::vector<tensor>> correct(n, std::vector<tensor>(k, tensor({1, 1})));

    for(int i = 0; i < n; i++){
        int val = rand() & ((1<<(k-1))-1);
        int val2 = rand() & ((1<<(k-1))-1);
        int valc = val+val2;
        
        for(int j = 0; j < k; j++){
            input[i][j][0] = val & 1;
            input[i][j][1] = val2 & 1;
            correct[i][j][0] = valc & 1;

            val >>= 1;
            val2 >>= 1;
            valc >>= 1;
        }
    }

    myNetwork.learn(input, correct);

}
#endif

TEST_F(NetworkTest, NetworkSave) {
    Network myNetwork(inputDefObject({1,2}), {layerDefObject({1,20}, layers::recursive, activations::ReLU)}, outputDefObject({1,1}, layers::perceptron, activations::Sigmoid), 1);
    
    myNetwork.save("savetest");

    Network myNetwork2("savetest");
}

TEST_F(NetworkTest, NetworkCheckpoint) {
    Network myNetwork(inputDefObject({1,2}), {layerDefObject({1,20}, layers::recursive, activations::ReLU)}, outputDefObject({1,1}, layers::perceptron, activations::Sigmoid), 1);
    
    myNetwork.save("savetest", save::checkpoint);

    Network myNetwork2("savetest");

    // some sort of comparison between network 1 and 2

    tensor input({1,2});
    input.cpy({0,1});
    tensor correct({1,1});
    correct.cpy({1});

    // make sure learn is the same
    myNetwork.learn({{input}}, {{correct}});
    std::vector<tensor> output = myNetwork.getOutput();

    myNetwork2.learn({{input}}, {{correct}});
    std::vector<tensor> output2 = myNetwork2.getOutput();

    ASSERT_TRUE(output[0] == output2[0]);


    // make sure forward in still correct, showing whole sequence is the same
    myNetwork.forward({input});
    output = myNetwork.getOutput();

    myNetwork2.forward({input});
    output2 = myNetwork2.getOutput();

    ASSERT_TRUE(output[0] == output2[0]);
}


}
}
}




