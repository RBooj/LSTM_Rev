#include "ForgetGate.h"

/*
[Input]--------------|
                    (*)---------------|
[Input weight]-------|                |
                                     (+)---------|
[Hidden State]--------------|         |          |
                           (*)--------|         (+)------------[Output]
[Hidden state weight]-------|                    |
                                                 |
[Bias]-------------------------------------------|
*/

// Feedforward is how the ForgetGate generates an output value
// Equation: _forget_result = sig([input * input_weight] + [hidden_state * hidden_weight] + bias)
void ForgetGate::feedforward()
{
    // Perform the calculation, and use it to update the cell state
    // Gather values
    vector<double> input = get_input();
    vector<double> input_weight = get_input_weight();
    vector<double> hidden_state = get_hidden_state();
    vector<double> hidden_weight = get_state_weight();
    vector<double> bias = get_bias();
    int batch_size = get_batch_size();

    // Perform math

    /*
        Questions:
            Are the internal state vectors (input, hidden state, ect) all the same size?
                They must be the same size in order to be added element-wise
            What determines the dimensions of the internal state vectors?

    */

    // Input
    vector<double> weighted_input;
    for (size_t i = 0; i < batch_size; i++)
    {
        weighted_input.push_back(input.at(i) * input_weight.at(i));
    }

    // Hidden State
    vector<double> weighted_state;
    for (size_t i = 0; i < batch_size; i++)
    {
        weighted_state.push_back(hidden_state.at(i) * hidden_weight.at(i));
    }

    // Add bias
    vector<double> biased_sum;
    for (size_t i = 0; i < batch_size; i++)
    {
        biased_sum.push_back(weighted_input.at(i) + weighted_state.at(i) + bias.at(i));
    }

    // Calculate activation function(sigmoid)
    for (size_t i = 0; i < batch_size; i++)
    {
        _forget_result.push_back(sigmoid(biased_sum.at(i)));
    }

    // Update Cell state
}