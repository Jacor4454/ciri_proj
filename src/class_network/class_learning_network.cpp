#include "class_learning_network.h"


learningNetwork::learningNetwork(inputDefObject i, std::vector<layerDefObject> ls, outputDefObject o, int maxItt = 1){
    // get network size and expand/reserve
    N = ls.size()+1;
    currMaxItt = maxItt;
    lastItt = 0;
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
        layers[i]->setAcc(ls[i].getAccTyp());
    }
    dimss[N] = o.getDims();
    layers[N-1] = network::getLayer(o.getLyrTyp(), dimss[N-1], dimss[N]);
    layers[N-1]->setAcc(o.getAccTyp());

    // make tensors for passing
    for(auto& v : dimss)
        for(int i = 0; i < currMaxItt+1; i++)
            tss[i].push_back(tensor(v));
    
    for(int i = 0; i < N+1; i++)
        tss[0][i].sMult(tss[0][i], 0);

    // make tensors for learning
    invts.reserve(N+1);
    for(auto& v : dimss)
        invts.push_back(tensor(v));
}

void learningNetwork::resizeItt(int newCurr){
    // check input (no throw here)
    if(newCurr <= currMaxItt)
        return;
    
    // get network size and expand/reserve
    tss.resize(newCurr+1);
    for(int i = currMaxItt+1; i < newCurr+1; i++)
        tss[i].reserve(N+1);
    
    // make tensors for passing
    for(auto& v : dimss)
        for(int i = currMaxItt+1; i < newCurr+1; i++)
            tss[i].push_back(tensor(v));

    currMaxItt = newCurr;
}

void learningNetwork::forward(const std::vector<tensor>& input){
    int itts = input.size();
    lastItt = itts;

    if(itts > currMaxItt)
        resizeItt(itts);

    for(int i = 0; i < N; i++)
        layers[i]->clear();

    for(int it = 0; it < input.size(); it++){
        tss[it+1][0].cpy(input[it]);

        for(int i = 0; i < N; i++){
            layers[i]->forward(tss[it+1][i+1], tss[it+1][i]);
        }
    }
}

std::vector<tensor> learningNetwork::getOutput(){
    std::vector<tensor> output(lastItt, tensor({}));

    for(int i = 0; i < lastItt; i++)
        output[i].cpy(tss[i+1][N]);

    return output;
}

void learningNetwork::backward(const std::vector<tensor>& correct){
    int itts = correct.size();

    if(itts != lastItt)
        // will be a resizer but for now is throw
        throw std::runtime_error("backward: itteration does not match previous itteration");

    // zero layers carry (if applicable)

    // itterated backward through time
    float loss = 0;
    for(int it = itts-1; it >= 0; it--){
        // differentiate the top
        loss += tss[it+1][N].loss(correct[it], errors::MSE);
        tss[it+1][N].gradient(invts[N], correct[it], errors::MSE);

        for(int i = N-1; i >= 0; i--)
            layers[i]->backward(invts[i], tss[it+1][i], invts[i+1], tss[it][i+1], tss[it+1][i+1]);
    }
    // std::cout << "loss: " << loss << "\n";
    
    // learn all layers
    for(int i = 0; i < N; i++)
        layers[i]->learn();
}

void learningNetwork::learn(const std::vector<std::vector<tensor>>& input, const std::vector<std::vector<tensor>>& correct){
    if(input.size() != correct.size())
        throw std::runtime_error("input and lable have different ammounts of data");

    int n = input.size();

    int per = 0;
    for(int i = 0; i < n; i++){
        forward(input[i]);
        backward(correct[i]);
        
        bool corr = true;
        std::vector<tensor> out = getOutput();
        for(int j = 0; j < out.size(); j++)
            if(round(out[j][0]) != correct[i][j][0])
                corr = false;

        if(corr)
            per++;
        if(i % 100 == 99){
            std::cout << "acc: " << per << "%\n";
            per = 0;
        }
    }
}


