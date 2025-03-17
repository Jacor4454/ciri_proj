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
    }

    ~TensorTest() override {
        // You can do clean-up work that doesn't throw exceptions here.
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
    tensor<float> myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 10);
}
TEST_F(TensorTest, MakeTensorSize2DTest) {
    std::vector<int> dims = {3,4};
    tensor<int> myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 12);
}
TEST_F(TensorTest, MakeTensorSize3DTest) {
    std::vector<int> dims = {3,4,5};
    tensor<double> myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 60);
}


}
}
}

