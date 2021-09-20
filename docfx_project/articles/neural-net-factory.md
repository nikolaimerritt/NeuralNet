# Neural Net Factory
The `NeuralNetFactory` class makes it easy to initialise a `NeuralNet` for most practical purposes.

## Starting From Scratch ##

### Layer Structure
First, decide on the structure you want your layers to have. This is done by creating an `IList<NeuralLayerConfig>`. For example, if you want to have:
 - an input layer of size 30
 - a hidden layer of size 20, using ReLU activation
 - a hidden layer of size 15, using ReLU activation
 - an output layer of size 10, using no activation at all

then create the following list:

```cs
IList<NeuralLayerConfig> layerStructure = new List<NeuralLayerConfig>() 
{
    new InputLayer(size: 30),
    new HiddenLayer(size: 20, activation: new ReluActivation()),
    new HiddenLayer(size: 15, activation: new ReluActivation()),
    new OutputLayer(size: 10, activation: new IdentityActivation())
};
```

### Gradient Descender
Next, decide on which method of gradient descent you wish to use. Adam gradient descent with default arguments is reccomended for most training data sets:

```cs
GradientDescender gradientDescender = new AdamGradientDescender();
```

### Cost Function
Next, decide on which cost function you wish to use. This will depend heavily on what kind of data you wish to learn. Here, I have chosen mean-squared-error cost:

```cs
CostFunction cost = new MSECost();
```

## Creating your Neural Network
There are various ways of best initializing a Neural Network, depending on which activators it will use and what training data it will be learning.

### Optimising for ReLU learning
If your layer structure mostly uses ReLU activation (such as in the example above), you can create a Neural Net with random weights and biases that is optimised for learning with ReLU:

```cs
NeuralNet neuralNet = NeuralNetFactory.OptimisedForRelu(layerStructure, gradientDescender, cost);
```
### Optimising for tanh learning
If your layer structure mostly uses tanh activation, you can create a NeuralNet with random weights and biases that is optimised for learning with tanh:

```cs
NeuralNet neuralNet = NeuralNetFactory.OptimisedForTanh(layerStructure, gradientDescender, cost);
```

### Optimising for learning a particular data set
A more recent option is to create a NeuralNet with random weights and biases that is optimised for learning some particular training data. When optimising, this will take into account the NeuralNet's layer structure and activators. However, it will not take into account the gradient descent method or cost function used.


```cs
IEnumerable<(Vector<double> input, Vector<double> desiredOutput)> trainingData = ... ;
NeuralNet neuralNet = NeuralNetFactory.OptimisedForTrainingData(layerStructure, trainingData, gradientDescender, cost);
```

Since the method used in `NeuralNetFactory.OptimisedForTrainingData()` only uses the input vectors in the training data, there is an overload that simply takes the training input vectors:

```cs
IEnumerable<Vector<double>> trainingInputs = ... ;
NeuralNet neuralNet = NeuralNetFactory.OptimisedForTrainingData(layerStructure, trainingInputs, gradientDescender, cost);
```

## Reading a `NeuralNet` from a directory
If you have previously written a NeuralNet to a directory using `NeuralNet.WriteToDirectory()`, then `NeuralNetFactory.ReadFromDirectory()` will load the NeuralNet back in. `NeuralNet.ReadFromDirectory()` returns a new NeuralNet with identical:
 - `Parameter` (so, identical weights, biases and layer structure)
 - `Activator`s applied to their corresponding layers
 - `GradientDescender`
 - `CostFunction`
to the original NeuralNet that was written to the directory.

For example:
```cs
NeuralNet neuralNet = ... ;
neuralNet.GradientDescent(...);
neuralNet.WriteToDirectory("../../neural-net-state");

NeuralNet read = NeuralNetFactory.ReadFromDirectory("../../neural-net-state");
// `read` now has the same internal state as `neuralNet`.
```