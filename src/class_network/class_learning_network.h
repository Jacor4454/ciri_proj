#ifndef CLASS_LEARNING_NETWORK_H
#define CLASS_LEARNING_NETWORK_H

#include "class_network.h"

class learningNetwork : public network {
    private:
    std::vector<tensor> invts;

    public:
    learningNetwork(inputDefObject, std::vector<layerDefObject>, outputDefObject);
    // ~learningNetwork();
    void backward(const tensor& correct);
};


#endif