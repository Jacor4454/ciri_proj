#include "class_learning_network.h"


learningNetwork::learningNetwork(inputDefObject i, std::vector<layerDefObject> ls, outputDefObject o, int maxItt = 1){
    // get network size and expand/reserve
    N = ls.size()+1;
    currMaxItt = maxItt;
    layers.resize(N, nullptr);
    dimss.resize(N+1);
    tss.resize(currMaxItt+1);
    for(int i = 0; i < currMaxItt+1; i++)
        tss[i].reserve(N+1);

    // assemble network
    dimss[0] = i.getDims();
    for(int i = 0; i < N-1; i++){
        dimss[i+1] = ls[i].getDims();
        layers[i] = network::getLayer(ls[i].getLyrTyp(), dimss[i], dimss[i+1]);
    }
    dimss[N] = o.getDims();
    layers[N-1] = network::getLayer(o.getLyrTyp(), dimss[N-1], dimss[N]);

    // make tensors for passing
    for(auto& v : dimss)
        for(int i = 0; i < currMaxItt+1; i++)
            tss[i].push_back(tensor(v));

    // make tensors for learning
    invts.reserve(N+1);
    for(auto& v : dimss)
        invts.push_back(tensor(v));
}


void learningNetwork::forward(const std::vector<tensor>& input){
    int itts = input.size();

    if(itts > currMaxItt)
        // will be a resizer but for now is throw
        throw std::runtime_error("itteration excedes allocated itteration limit");
    
    for(int it = 0; it < input.size(); it++){
        tss[it+1][0].cpy(input[it]);

        for(int i = 0; i < N; i++){
            layers[i]->forward(tss[it+1][i+1], tss[it+1][i]);
        }

        // do some result handling
    }
}


void learningNetwork::backward(const std::vector<tensor>& correct){
    int itts = correct.size();

    if(itts > currMaxItt)
        // will be a resizer but for now is throw
        throw std::runtime_error("itteration excedes allocated itteration limit");

    // zero layers carry (if applicable)

    // itterated backward through time
    for(int it = itts-1; it >= 0; it--){
        // differentiate the top
        tss[it][N].gradient(invts[N], correct[it], errors::MSE);

        for(int i = N-1; i >= 0; i--){
            layers[i]->backward(invts[i], tss[it+1][i], invts[i+1], tss[it][i+1], tss[it+1][i+1]);
        }
    }

    // learn all layers
}




