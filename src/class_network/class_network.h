#ifndef CLASS_NETWORK_H
#define CLASS_NETWORK_H

#include "../class_layer/class_layer.h"

class inputDefObject {
    private:
    std::vector<int> inputDims;
    public:
    inputDefObject(std::vector<int> i):inputDims(i){}
    std::vector<int> getDims(){return inputDims;}
};
class layerDefObject {
    private:
    std::vector<int> outputDims;
    public:
    layerDefObject(std::vector<int> o):outputDims(o){}
    std::vector<int> getDims(){return outputDims;}
};
class outputDefObject {
    private:
    std::vector<int> outputDims;
    public:
    outputDefObject(std::vector<int> o):outputDims(o){};
    std::vector<int> getDims(){return outputDims;}
};


class network{
    std::vector<BaseLayer*> layers;
    std::vector<std::vector<int>> dimss;
    int N = 0;
    network(inputDefObject, std::vector<layerDefObject>, outputDefObject);
};


#endif