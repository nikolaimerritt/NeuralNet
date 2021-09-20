using NeuralNetLearning.Maths.Activations;

namespace NeuralNetLearning.LayerConfig
{
    public abstract record NeuralLayerConfig
    {
        public int Size { get; protected init; }
    }
}
