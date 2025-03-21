#include <gtest/gtest.h>

#include "../../src/class_tensor/class_tensor.h"

// #define TENSORBENCHMARK

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
        tensor::threads_setWorkers(4);
        tensor::threads_initaliseThreads();
    }

    ~TensorTest() override {
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
TEST_F(TensorTest, MakeTensorSize1D) {
    std::vector<int> dims = {10};
    tensor myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 10);
}
TEST_F(TensorTest, MakeTensorSize2D) {
    std::vector<int> dims = {3,4};
    tensor myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 12);
}
TEST_F(TensorTest, MakeTensorSize3D) {
    std::vector<int> dims = {3,4,5};
    tensor myTensor(dims);
    EXPECT_EQ(myTensor.getN(), 60);
}
TEST_F(TensorTest, ThreadError) {
    EXPECT_THROW(tensor::threads_initaliseThreads(), std::runtime_error);
    tensor::threads_killThreads();
    EXPECT_THROW(tensor::threads_killThreads(), std::runtime_error);
    tensor::threads_setWorkers(8);
    tensor::threads_initaliseThreads();
    EXPECT_EQ(tensor::threads_getActiveWorkers(), 8);
    EXPECT_THROW(tensor::threads_setWorkers(4), std::runtime_error);
}
TEST_F(TensorTest, TensorAdd) {
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
    EXPECT_TRUE(myTensor4 == myTensor3);
}
TEST_F(TensorTest, TensorSMult) {
    std::vector<int> dims = {3,4,5};
    tensor myTensor1(dims);
    tensor myTensor2(dims);
    tensor myTensor3(dims);
    for(int i = 0; i < 60; i++){
        myTensor1.getContents()[i] = i;
        myTensor2.getContents()[i] = i*2;
        myTensor3.getContents()[i] = i*i*2;
    }
    tensor myTensor4 = myTensor1 * myTensor2;
    EXPECT_TRUE(myTensor4 == myTensor3);
}
TEST_F(TensorTest, TensorMult) {
    int n = 2;
    int m = 3;
    int k = 4;
    std::vector<int> dims1({n, k});
    std::vector<int> dims2({k, m});
    std::vector<int> dims3({n, m});
    tensor myTensor1(dims1);
    for(int i = 0; i < 8; i++)
        myTensor1.getContents()[i] = i+1;
    tensor myTensor2(dims2);
    for(int i = 0; i < 12; i++)
        myTensor2.getContents()[i] = i+1;
    tensor myTensor3(dims3);
    std::vector<float> data = {70,80,90,158,184,210};
    for(int i = 0; i < 6; i++)
        myTensor3.getContents()[i] = data[i];
    tensor myTensor4 = myTensor1 ^ myTensor2;
    EXPECT_TRUE(myTensor4 == myTensor3);
}
TEST_F(TensorTest, Tensor3DMult) {
    int n = 2;
    int m = 3;
    int k = 4;
    int Ex = 2;
    std::vector<int> dims1({Ex, n, k});
    std::vector<int> dims2({Ex, k, m});
    std::vector<int> dims3({Ex, n, m});
    tensor myTensor1(dims1);
    for(int i = 0; i < 16; i++)
        myTensor1.getContents()[i] = i+1;
    tensor myTensor2(dims2);
    for(int i = 0; i < 24; i++)
        myTensor2.getContents()[i] = i+1;
    tensor myTensor3(dims3);
    std::vector<float> data = {70,80,90,158,184,210,750,792,834,1030,1088,1146};
    for(int i = 0; i < 12; i++)
        myTensor3.getContents()[i] = data[i];
    tensor myTensor4 = myTensor1 ^ myTensor2;
    EXPECT_TRUE(myTensor4 == myTensor3);
}
TEST_F(TensorTest, TensorMultN) {
    int n = 3;
    int m = 1;
    int k = 4;
    std::vector<int> dims1({n, k});
    std::vector<int> dims2({k, m});
    std::vector<int> dims3({n, m});
    tensor myTensor1(dims1);
    for(int i = 0; i < 12; i++)
        myTensor1.getContents()[i] = i+1;
    tensor myTensor2(dims2);
    for(int i = 0; i < 4; i++)
        myTensor2.getContents()[i] = i+1;
    tensor myTensor3(dims3);
    std::vector<float> data = {30,70,110};
    for(int i = 0; i < 3; i++)
        myTensor3.getContents()[i] = data[i];
    tensor myTensor4 = myTensor1 ^ myTensor2;
    EXPECT_TRUE(myTensor4 == myTensor3);
}
TEST_F(TensorTest, Tensor3DMultN) {
    int n = 3;
    int m = 1;
    int k = 4;
    int Ex = 2;
    std::vector<int> dims1({Ex, n, k});
    std::vector<int> dims2({Ex, k, m});
    std::vector<int> dims3({Ex, n, m});
    tensor myTensor1(dims1);
    for(int i = 0; i < 24; i++)
        myTensor1.getContents()[i] = i+1;
    tensor myTensor2(dims2);
    for(int i = 0; i < 8; i++)
        myTensor2.getContents()[i] = i+1;
    tensor myTensor3(dims3);
    std::vector<float> data = {30,70,110,382,486,590};
    for(int i = 0; i < 6; i++)
        myTensor3.getContents()[i] = data[i];
    tensor myTensor4 = myTensor1 ^ myTensor2;
    EXPECT_TRUE(myTensor4 == myTensor3);
}
TEST_F(TensorTest, TensorMultErrors) {
    int n = 3;
    int m = 1;
    int k = 4;
    int Ex = 2;

    // k dim different
    std::vector<int> dims1({Ex, n, k+1});
    std::vector<int> dims2({Ex, k, m});
    tensor myTensor1(dims1);
    tensor myTensor2(dims2);
    EXPECT_THROW(myTensor1^myTensor2, std::runtime_error);
    
    // mismatched dim length
    std::vector<int> dims3({n, k});
    tensor myTensor3(dims3);
    EXPECT_THROW(myTensor3^myTensor2, std::runtime_error);
    
    // length < 2
    std::vector<int> dims4({n});
    std::vector<int> dims5({m});
    tensor myTensor4(dims4);
    tensor myTensor5(dims5);
    EXPECT_THROW(myTensor4^myTensor5, std::runtime_error);

    // incorect precede dim
    std::vector<int> dims6({Ex+1, n, k});
    tensor myTensor6(dims6);
    EXPECT_THROW(myTensor6^myTensor2, std::runtime_error);
    
    // stopped threads
    std::vector<int> dims7({Ex, n, k});
    tensor myTensor7(dims7);
    tensor::threads_killThreads();
    EXPECT_THROW(myTensor7^myTensor2, std::runtime_error);
    tensor::threads_initaliseThreads();
    myTensor7^myTensor2;

    
    // output wrong size
    std::vector<int> dims8({Ex, n+1, m});
    tensor myTensor8(dims8);
    EXPECT_THROW(myTensor7.mult(myTensor8, myTensor2), std::runtime_error);
    std::vector<int> dims9({Ex+1, n, m});
    tensor myTensor9(dims9);
    EXPECT_THROW(myTensor7.mult(myTensor9, myTensor2), std::runtime_error);
    std::vector<int> dims10({Ex, n, m});
    tensor myTensor10(dims10);
    myTensor7.mult(myTensor10, myTensor2);

}
TEST_F(TensorTest, TensorAddErrors) {
    int n = 3;
    int m = 1;
    int k = 4;
    int Ex = 2;

    // k dim different
    std::vector<int> dims1({Ex, k, n, m});
    std::vector<int> dims2({Ex, k, n, m+1});
    tensor myTensor1(dims1);
    tensor myTensor2(dims2);
    EXPECT_THROW(myTensor1+myTensor2, std::runtime_error);
    
    std::vector<int> dims3({Ex, k+1, n, m});
    tensor myTensor3(dims3);
    EXPECT_THROW(myTensor1+myTensor3, std::runtime_error);
    
    std::vector<int> dims4({Ex, k, n, m});
    tensor myTensor4(dims4);
    myTensor1+myTensor4;

    std::vector<int> dims5({Ex, k, n, m+1});
    tensor myTensor5(dims5);
    EXPECT_THROW(myTensor1.add(myTensor5, myTensor4), std::runtime_error);
    
    std::vector<int> dims6({Ex+1, k, n, m});
    tensor myTensor6(dims6);
    EXPECT_THROW(myTensor1.add(myTensor6, myTensor4), std::runtime_error);

    EXPECT_THROW(myTensor1.add(myTensor6, 12), std::runtime_error);    

    
    std::vector<int> dims7({Ex, k, n, m});
    tensor myTensor7(dims7);
    myTensor1.add(myTensor7, 12);
    myTensor1.add(myTensor7, myTensor4);
}
TEST_F(TensorTest, TensorSMultErrors) {
    int n = 3;
    int m = 1;
    int k = 4;
    int Ex = 2;

    // k dim different
    std::vector<int> dims1({Ex, k, n, m});
    std::vector<int> dims2({Ex, k, n, m+1});
    tensor myTensor1(dims1);
    tensor myTensor2(dims2);
    EXPECT_THROW(myTensor1*myTensor2, std::runtime_error);
    
    std::vector<int> dims3({Ex, k+1, n, m});
    tensor myTensor3(dims3);
    EXPECT_THROW(myTensor1*myTensor3, std::runtime_error);
    
    std::vector<int> dims4({Ex, k, n, m});
    tensor myTensor4(dims4);
    myTensor1*myTensor4;

    std::vector<int> dims5({Ex, k, n, m+1});
    tensor myTensor5(dims5);
    EXPECT_THROW(myTensor1.sMult(myTensor5, myTensor4), std::runtime_error);
    
    std::vector<int> dims6({Ex+1, k, n, m});
    tensor myTensor6(dims6);
    EXPECT_THROW(myTensor1.sMult(myTensor6, myTensor4), std::runtime_error);

    EXPECT_THROW(myTensor1.sMult(myTensor6, 12), std::runtime_error);    

    
    std::vector<int> dims7({Ex, k, n, m});
    tensor myTensor7(dims7);
    myTensor1.sMult(myTensor7, 12);
    myTensor1.sMult(myTensor7, myTensor4);
}
TEST_F(TensorTest, TensorKAdd) {
    float k = 2;
    std::vector<int> dims = {3,4,5};
    tensor myTensor1(dims);
    tensor myTensor2(dims);
    for(int i = 0; i < 60; i++){
        myTensor1.getContents()[i] = i;
        myTensor2.getContents()[i] = i + k;
    }
    tensor myTensor3 = myTensor1 + k;
    EXPECT_TRUE(myTensor2 == myTensor3);
}
TEST_F(TensorTest, TensorKMult) {
    float k = 2;
    std::vector<int> dims = {3,4,5};
    tensor myTensor1(dims);
    tensor myTensor2(dims);
    for(int i = 0; i < 60; i++){
        myTensor1.getContents()[i] = i;
        myTensor2.getContents()[i] = i * k;
    }
    tensor myTensor3 = myTensor1 * k;
    EXPECT_TRUE(myTensor2 == myTensor3);
}

#ifdef TENSORBENCHMARK
void benchmarkHelper(){
    int n = 1;
    int m = 150;
    int k = 784;
    std::vector<int> dims1({n, k});
    std::vector<int> dims2({k, m});
    std::vector<int> dims3({n, m});
    tensor myTensor1(dims1);
    tensor myTensor2(dims2);
    long long total = 0;
    int tests = 500;
    for(int i = 0; i < tests; i++){
        auto start = std::chrono::high_resolution_clock::now();
        tensor myTensor4 = myTensor1 ^ myTensor2;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total += duration.count();
    }
    std::cout << tensor::threads_getActiveWorkers() << " threads took: " << total/tests << " nanoseconds on avg over " << tests << " loops" << std::endl;
}
TEST_F(TensorTest, TensorThreadBenchmark){
    std::vector<int> tests = {1,2,4,8,16};
    for(int i : tests){
        tensor::threads_killThreads();
        tensor::threads_setWorkers(i);
        tensor::threads_initaliseThreads();
        benchmarkHelper();
    }
}
TEST_F(TensorTest, TensorAssignmentBenchmark){
    std::cout << "Using: " << tensor::threads_getActiveWorkers() << " threads" << std::endl;
    int n = 1;
    int m = 150;
    int k = 784;
    std::vector<int> dims1({n, k});
    std::vector<int> dims2({k, m});
    std::vector<int> dims3({n, m});
    tensor myTensor1(dims1);
    tensor myTensor2(dims2);
    tensor myTensor3(dims3);

    long long total = 0;
    int tests = 500;
    for(int i = 0; i < tests; i++){
        auto start = std::chrono::high_resolution_clock::now();
        tensor myTensor4 = myTensor1 ^ myTensor2;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total += duration.count();
    }
    std::cout << "operator took: " << total/tests << " nanoseconds on avg over " << tests << " loops" << std::endl;
    
    total = 0;
    for(int i = 0; i < tests; i++){
        auto start = std::chrono::high_resolution_clock::now();
        myTensor1.mult(myTensor3, myTensor2);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        total += duration.count();
    }
    std::cout << "prealloc took: " << total/tests << " nanoseconds on avg over " << tests << " loops" << std::endl;
}
#endif


}
}
}

