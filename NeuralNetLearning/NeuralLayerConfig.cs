using System;

namespace NeuralNetLearning
{
    public abstract record NeuralLayer
    {
        public int LayerSize { get; init; }
    }

    public record InputLayer : NeuralLayer 
    {
        public InputLayer(int layerSize)
            => LayerSize = layerSize;
    }

    public record HiddenLayer : NeuralLayer
    {
        public Activation Activator { get; init; }
        public HiddenLayer(int layerSize, Activation activator)
        {
            LayerSize = layerSize;
            Activator = activator;
        }
    }

    public record OutputLayer : NeuralLayer
    {
        public Activation Activator { get; init; }
        public OutputLayer(int layerSize, Activation activator)
        {
            LayerSize = layerSize;
            Activator = activator;
        }
    }
}
