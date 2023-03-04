#include "LSTMGate.h"

// No constructors for base class
// Define Public functions to be inherited

// Getters
// =================================================
vector<double> &LSTMGate::get_input()
{
    return _internal_state._input;
}

vector<double> &LSTMGate::get_hidden_state()
{
    return _internal_state._hidden_state;
}

vector<double> &LSTMGate::get_cell_state()
{
    return _internal_state._cell_state;
}

vector<double> &LSTMGate::get_output()
{
    return _internal_state._output;
}

vector<double> &LSTMGate::get_input_weight()
{
    return _internal_state._input_weight;
}

vector<double> &LSTMGate::get_state_weight()
{
    return _internal_state._state_weight;
}

vector<double> &LSTMGate::get_bias()
{
    return _internal_state._bias;
}
// =================================================

// Increment tunable parameters function
void LSTMGate::increment_tunable(int feature, int index, double inc_amt)
{
    switch (feature)
    {
    case BIAS:
        _internal_state._bias.at(index) += inc_amt;
        break;
    case STATE:
        _internal_state._state_weight.at(index) += inc_amt;
        break;
    case INPUT:
        _internal_state._input_weight.at(index) += inc_amt;
        break;
    default:
        throw "Feature did not match any allowed value";
        break;
    }
}
