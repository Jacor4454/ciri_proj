#include "class_network.h"


network::network(inputDefObject i, std::vector<layerDefObject> ls, outputDefObject o){
    N = ls.size()+1;
    layers.resize(N);
    dimss.resize(N+1);
    dimss[0] = i.getDims();
    for(int i = 0; i < N; i++){
        // passing input dims to layers
        // how tho
    }
}


