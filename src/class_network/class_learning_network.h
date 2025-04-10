#ifndef CLASS_LEARNING_NETWORK_H
#define CLASS_LEARNING_NETWORK_H

#include "class_network.h"

class learningNetwork {
    private:
    std::vector<BaseLayer*> layers;
    std::vector<std::vector<int>> dimss;
    int N = 0;
    int currMaxItt, lastItt;
    std::vector<std::vector<tensor>> tss;
    std::vector<tensor> invts;

    void resizeItt(int newCurr);

    public:
    learningNetwork(inputDefObject, std::vector<layerDefObject>, outputDefObject, int);
    // ~learningNetwork();
    void forward(const std::vector<tensor>& input);
    std::vector<tensor> getOutput();
    void backward(const std::vector<tensor>& correct);

    void learn(const std::vector<std::vector<tensor>>& input, const std::vector<std::vector<tensor>>& correct);
};


#endif