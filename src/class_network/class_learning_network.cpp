#include "class_learning_network.h"


learningNetwork::learningNetwork(inputDefObject i, std::vector<layerDefObject> ls, outputDefObject o):network(i, ls, o){
    // make tensors for learning
    invts.reserve(N+1);
    for(auto& v : dimss)
        invts.push_back(tensor(v));
}


void learningNetwork::backward(const tensor& correct){
    ts[N].gradient(invts[N], correct, errors::MSE);

    for(int i = N-1; i >= 0; i--){
        layers[i]->backward(invts[i], ts[i], invts[i+1], ts[i+1], ts[i+1]);//first ts[i+1] is back in time
    }
}




