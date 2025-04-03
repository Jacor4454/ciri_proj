#include "class_network.h"

// netowrk construction helper to get the network from template
BaseLayer* network::getLayer(layers::layerTypes l, std::vector<int>& in, std::vector<int>& out){
    BaseLayer* output = nullptr;
    switch(l){
        case layers::perceptron:
            output = new perceptron(in, out);
            break;
        case layers::recursive:
            output = new recursive(in, out);
            break;
        default:
            throw std::runtime_error("layer not implemented in network constructor");
    }
    return output;
}

network::network(inputDefObject i, std::vector<layerDefObject> ls, outputDefObject o){
    // get network size and expand/reserve
    N = ls.size()+1;
    layers.resize(N, nullptr);
    dimss.resize(N+1);
    ts.reserve(N+1);

    // assemble network
    dimss[0] = i.getDims();
    for(int i = 0; i < N-1; i++){
        dimss[i+1] = ls[i].getDims();
        layers[i] = getLayer(ls[i].getLyrTyp(), dimss[i], dimss[i+1]);
    }
    dimss[N] = o.getDims();
    layers[N-1] = getLayer(o.getLyrTyp(), dimss[N-1], dimss[N]);

    // make tensors for passing
    for(auto& v : dimss)
        ts.push_back(tensor(v));
}

network::~network(){
    for(int i = 0; i < N; i++)
        if(layers[i] != nullptr)
            delete layers[i];
}

void network::forward(const std::vector<tensor>& input){
    for(int it = 0; it < input.size(); it++){
        ts[0].cpy(input[it]);

        for(int i = 0; i < N; i++){
            layers[i]->forward(ts[i+1], ts[i]);
        }

        // do some result handling
    }
}

