using System;

namespace NeuralNetLearning
{
    public abstract record NeuralLayerConfig
    {
        public int LayerSize { get; init; }
    }

    public record InputLayer : NeuralLayerConfig 
    {
        public InputLayer(int layerSize)
            => LayerSize = layerSize;
    }

    public record HiddenLayer : NeuralLayerConfig
    {
        public Activation Activator { get; init; }
        public HiddenLayer(int layerSize, Activation activator)
        {
            LayerSize = layerSize;
            Activator = activator;
        }
    }

    public record OutputLayer : NeuralLayerConfig
    {
        public Activation Activator { get; init; }
        public OutputLayer(int layerSize, Activation activator)
        {
            LayerSize = layerSize;
            Activator = activator;
        }
    }
}
