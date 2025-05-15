#include "src/class_network/class_learning_network.h"


int main(){
    // start engine
    tensor::threads_setWorkers(4);
    tensor::threads_initaliseThreads();
    log::open("log.txt");

    // load data
    int n = 5000; // no. datapoints
    std::vector<std::vector<tensor>> inputs(n, std::vector<tensor>(1, tensor({1,784})));
    std::vector<std::vector<tensor>> correct(n, std::vector<tensor>(1, tensor({1,10})));
    
    std::ifstream f("grids.txt");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < 784; j++){
            f >> inputs[i][0][j];
            inputs[i][0][j] /= 255;
        }
    }
    f.close();
    
    std::ifstream f2("labels.txt");
    for(int i = 0; i < n; i++){
        int ind;
        f2 >> ind;
        correct[i][0].set(0.0);
        correct[i][0][ind] = 1;
    }
    f2.close();

    // define network
    Network MNISTNetwork(inputDefObject({1,784}), {layerDefObject({1,25}, layers::perceptron, activations::ReLU, new AdagradLearnerSelector(0.01))}, outputDefObject({1,10}, layers::perceptron, activations::Sigmoid, errors::MSE, new AdagradLearnerSelector(0.01)), 1, 1231);

    // learn
    for(int i = 0; i < 2; i++){
        MNISTNetwork.learn(inputs, correct);
    }

    // deconstruct
    tensor::threads_killThreads();

}
