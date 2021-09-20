# Neural Net
A `NeuralNet` takes a `Vector<double>` as input and estimates the appropriate `Vector<double>` to output. By giving `NeuralNet` a training data set, consisting of pairs of sample inputs with the correct output, the `NeuralNet` will learn to produce more and more accurate outputs.

## Creating a Blank NeuralNet
`NeuralNetFactory` is used to create "blank" `NeuralNet`s that are ready to learn. You will then need to run `NeuralNet.Fit(...)` to learn your data.

## Preparing Training Data for a NeuralNet
The NeuralNet will learn how to produce the right output for a given input. Thus, you will need to prepare your data as a list of inputs paired with the corresponding output you want the NeuralNet to produce. 

Because a `NeuralNet` is a mathematical object, each input and output will need to be a `Vector<double>`. Every input vector must have the same length, and every output vector must have the same length. The output vectors can have different lengths to the input vectors, however.
If you want your `NeuralNet` to input or output a single number, you must use `Vector<double>`s of length 1.


For example, here is how you prepare 10,000 training data for a `NeuralNet` to learn to produce `f(x) = x * x`, where `x` is a number between -50 and 50:
```cs
List<(Vector<double>, Vector<double>)> trainingData = new(capacity: 10000);
// Producing 10,000 input-output pairs:
for (int i = 0; i < 10000; i++)
{
    Vector<double> input = 50 * Vector<double>.Build.Random(length: 1); // input is a random vector with one random element between -50 and 50
    Vector<double> expectedOutput = input.PointwisePower(2);            // expectedOutput is input squared
    trainingData.Add((input, expectedOutput));
}
```

## Fitting a NeuralNet to Training Data
Once you have prepared your training 