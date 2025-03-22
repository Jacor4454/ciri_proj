#include <gtest/gtest.h>


// all test includes
#include "tensorTest.cpp"
#include "perceptronLayerTest.cpp"



int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
