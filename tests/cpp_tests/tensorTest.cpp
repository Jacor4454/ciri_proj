#include <gtest/gtest.h>

#include "../../src/class_tensor/class_tensor.h"

namespace my {
namespace project {
namespace {

// The fixture for testing class Foo.
class TensorTest : public testing::Test {
    protected:
    // You can remove any or all of the following functions if their bodies would
    // be empty.

    TensorTest() {
        // You can do set-up work for each test here.
        tensor::initaliseThreads();
    }

    ~TensorTest() override {
        // You can do clean-up work that doesn't throw exceptions here.
        tensor::killThreads();
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
TEST_F(TensorTest, MakeTensorSize1DTest) {
    std::vector<int> dims = {10};
    tensor myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 10);
}

TEST_F(TensorTest, MakeTensorSize2DTest) {
    std::vector<int> dims = {3,4};
    tensor myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 12);
}

TEST_F(TensorTest, MakeTensorSize3DTest) {
    std::vector<int> dims = {3,4,5};
    tensor myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 60);
}

TEST_F(TensorTest, TensorAddTest) {
    std::vector<int> dims = {3,4,5};
    tensor myTensor1(dims);
    tensor myTensor2(dims);
    tensor myTensor3(dims);
    for(int i = 0; i < 60; i++){
        myTensor1.getContents()[i] = i;
        myTensor2.getContents()[i] = i*2;
        myTensor3.getContents()[i] = i*3;
    }
    tensor myTensor4 = myTensor1 + myTensor2;
    EXPECT_EQ(myTensor4 == myTensor3, true);
}


}
}
}

