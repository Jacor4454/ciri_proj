#include "src/class_network/class_learning_network.h"


int main(){
    auto start = std::chrono::high_resolution_clock::now();

    // start engine
    tensor::threads_setWorkers(8);
    tensor::threads_initaliseThreads();
    log::open("log.txt");

    // load data
    int n = 5000; // no. datapoints
    learning_data data({1,784}, {1,10}, n, 1);

    std::ifstream f("grids.txt");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < 784; j++){
            f >> data.input[i][0][j];
            data.input[i][0][j] /= 255;
        }
    }
    f.close();
    
    std::ifstream f2("labels.txt");
    for(int i = 0; i < n; i++){
        int ind;
        f2 >> ind;
        data.correct[i][0].set(0.0);
        data.correct[i][0][ind] = 1;
    }
    f2.close();

    // define network
    Network MNISTNetwork(inputDefObject({1,784}), {layerDefObject({1,100}, layers::perceptron, activations::ReLU, new AdagradLearnerSelector(0.01))}, outputDefObject({1,10}, layers::perceptron, activations::Sigmoid, errors::MSE, new AdagradLearnerSelector(0.01)), 1, 1231);

    // learn
    data.setEpoch(5);
    data.setRand(true);
    MNISTNetwork.learn(data);

    // deconstruct
    tensor::threads_killThreads();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "execution time: " << std::chrono::duration_cast<std::chrono::microseconds>(end-start).count() << "\n";
}
