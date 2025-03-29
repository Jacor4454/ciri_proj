#include "class_learning_network.h"


learningNetwork::learningNetwork(inputDefObject i, std::vector<layerDefObject> ls, outputDefObject o):network(i, ls, o){
    // make tensors for learning
    invts.reserve(N+1);
    for(auto& v : dimss)
        invts.push_back(tensor(v));
}





