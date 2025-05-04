#ifndef CLASS_GRADIENTS_H
#define CLASS_GRADIENTS_H

#include <math.h>
#include <stdexcept>
#include "../class_tensor.h"

// error lables
namespace errors {
    typedef enum{
        SE, // Square Error
        MSE, // Mean Squared Error
        CE, // Cross-Entropy
    } errTypes;
}

#endif