using NeuralNetLearning.Maths.Activations;

namespace NeuralNetLearning.LayerConfig
{
    public record OutputLayer : NeuralLayerConfig
    {
        public Activation Activation { get; private init; }
        public OutputLayer(int size, Activation activation)
        {
            Size = size;
            Activation = activation;
        }
    }
}
