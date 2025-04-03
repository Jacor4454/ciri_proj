#ifndef CLASS_LEARNING_NETWORK_H
#define CLASS_LEARNING_NETWORK_H

#include "class_network.h"

class learningNetwork {
    private:
    std::vector<BaseLayer*> layers;
    std::vector<std::vector<int>> dimss;
    int N = 0;
    int currMaxItt;
    std::vector<std::vector<tensor>> tss;
    std::vector<tensor> invts;

    public:
    learningNetwork(inputDefObject, std::vector<layerDefObject>, outputDefObject, int);
    // ~learningNetwork();
    void forward(const std::vector<tensor>& input);
    void backward(const std::vector<tensor>& correct);
};


#endif