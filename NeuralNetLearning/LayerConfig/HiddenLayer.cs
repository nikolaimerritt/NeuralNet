using NeuralNetLearning.Maths.Activations;

namespace NeuralNetLearning.LayerConfig
{
    public record HiddenLayer : NeuralLayerConfig
    {
        public Activation Activation { get; private init; }
        public HiddenLayer(int size, Activation activation)
        {
            Size = size;
            Activation = activation;
        }
    }
}
