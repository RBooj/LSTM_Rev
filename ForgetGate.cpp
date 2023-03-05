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

    // Perform math

    /*
        Questions:
            Are the internal state vectors (input, hidden state, ect) all the same size?
            What determines the dimensions of the internal state vectors?

    */

    // Input
    for (size_t i = 0; i < input.size(); i++)
    {
        /* code */
    }

    // Hidden State
    for (size_t i = 0; i < hidden_state.size(); i++)
    {
        /* code */
    }

    // Add bias

    // Save output value(s)

    // Update Cell state
}