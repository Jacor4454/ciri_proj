#include <gtest/gtest.h>

#include "../../src/class_layer/learners/class_base.h"

namespace my {
namespace project {
namespace {

// The fixture for testing class Foo.
class LearnerTest : public testing::Test {
    protected:
    // You can remove any or all of the following functions if their bodies would
    // be empty.

    LearnerTest() {
        // You can do set-up work for each test here.
        tensor::threads_setWorkers(1);
        tensor::threads_initaliseThreads();
    }

    ~LearnerTest() override {
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
TEST_F(LearnerTest, alpha) {
    tensor weights({2,3});
    tensor dweights({2,3});
    tensor correct({2,3});

    weights.cpy({1,2,3,4,5,6});
    dweights.cpy({1,2,3,4,5,6});
    correct.cpy({0.9,1.8,2.7,3.6,4.5,5.4});
    
    BaseLearnerSelector* ls1 = new AlphaLearnerSelector(0.1);
    BaseLearner* l1 = ls1->construct(&weights);

    l1->backprop(dweights);
    l1->learn();

    for(int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(weights[i], correct[i]);
}

TEST_F(LearnerTest, adagrad) {
    tensor weights({2,3});
    tensor dweights({2,3});
    tensor correct({2,3});

    weights.cpy({1,2,3,4,5,6});
    dweights.cpy({1,2,3,4,5,6});
    correct.cpy({0.90000005,1.90000001,2.900000001,3.9,4.9,5.9});
    
    BaseLearnerSelector* ls1 = new AdagradLearnerSelector(0.1);
    BaseLearner* l1 = ls1->construct(&weights);

    l1->backprop(dweights);
    l1->learn();

    for(int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(weights[i], correct[i]);

    
    correct.cpy({0.8292893286,1.829289324,2.829289323,3.829289322,4.829289322,5.829289322});
    l1->learn();

    for(int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(weights[i], correct[i]);
}

TEST_F(LearnerTest, momentum) {
    tensor weights({2,3});
    tensor dweights({2,3});
    tensor correct({2,3});

    weights.cpy({1,2,3,4,5,6});
    dweights.cpy({1,2,3,4,5,6});
    correct.cpy({0.9,1.8,2.7,3.6,4.5,5.4});
    
    BaseLearnerSelector* ls1 = new MomentumLearnerSelector(0.1, 0.9);
    BaseLearner* l1 = ls1->construct(&weights);

    l1->backprop(dweights);
    l1->learn();

    for(int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(weights[i], correct[i]);

    
    correct.cpy({0.71,1.42,2.13,2.84,3.55,4.26});
    l1->learn();

    for(int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(weights[i], correct[i]);
}

TEST_F(LearnerTest, adam) {
    tensor weights({2,3});
    tensor dweights({2,3});
    tensor correct({2,3});

    weights.cpy({1,2,3,4,5,6});
    dweights.cpy({1,2,3,4,5,6});
    correct.cpy({0.9000000005,1.9,2.9,3.9,4.9,5.9});
    
    BaseLearnerSelector* ls1 = new AdamLearnerSelector(0.1, 0.9, 0.999);
    BaseLearner* l1 = ls1->construct(&weights);

    l1->backprop(dweights);
    l1->learn();

    for(int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(weights[i], correct[i]);

    correct.cpy({0.80000031,1.800000003,2.800000001,3.800000001,4.8,5.8});
    l1->learn();

    for(int i = 0; i < 6; i++)
        ASSERT_FLOAT_EQ(weights[i], correct[i]);
}

}
}
}

