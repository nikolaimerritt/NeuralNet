# Neural Layer Configuration
When creating a `NeuralNet` from `NeuralNetFactory`, the layer structure is set by passing an `IList<NeuralLayerConfig>`. This `IList<NeuralLayerConfig>` must begin with an `InputLayer` and end with an `OutputLayer`. For example,

```cs
IList<NeuralLayerConfig> layerStrucutre = new List<NeuralLayerConfig>() 
{
    new InputLayer(size: 100),
    new HiddenLayer(size: 70, activation: new ReluActivation(leak: 0.01)),
    new HiddenLayer(size: 30, activation: new ReluActivation(leak: 0.01)),
    new OutputLayer(size: 10, activation: new SoftmaxActivation())
};
...
NeuralNet net = NeuralNetFactory.OptimisedForRelu(layerStrucutre, ... );

Vector<double> inputVector = Vector<double>.Build.Dense(length: 100, value: 0); // length of input vector must match `size` supplied to `InputLayer`
Vector<double> outputVector = net.GetOutputVector(inputVector); // `outputVector` is of length 10, since 10 is the value of `size` supplied to `OutputLayer`
```

### Technical Note:
Note that the `IList<NeuralLayerConfig>` does not represent how `NeuralNet` works internally. In creating a `NeuralNet` from `NeuralNetFactory`, the `IList<NeuralLayerConfig>` is purely used to initialise a `Parameter` object of the right size and to supply the right `Activation`s to the `NeuralNet`. Once this is done, the `NeuralNet` uses the `Parameter` object and `Activation` array.

## NeuralLayerConfig
`NeuralLayerConfig` is an abstract record, and is the base record for 
    - `InputLayer`
    - `HiddenLayer`
    - `OutputLayer`

## InputLayer
`InputLayer` stores the size of the first (i.e. input) layer. Once the `NeuralNet` is created, `InputLayer.Size` will be the size of the input vectors accepted by the `NeuralNet`. 

## HiddenLayer
`HiddenLayer` stores the size and activator used by a hidden layer. These are intermediary layers. They are not required, but they give complexity to the Neural Network.

## OutputLayer
`OutputLayer` stores the size and activator used by the final (i.e. output) layer. Once the `NeuralNet` is created, `OutputLayer.Size` will be the size of the output vectors returned by the `NeuralNet`, and `OutputLayer.Activation` will be applied to these output vectors.