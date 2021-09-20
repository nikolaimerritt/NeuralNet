namespace NeuralNetLearning.LayerConfig
{
    public record InputLayer : NeuralLayerConfig
    {
        public InputLayer(int size)
            => Size = size;
    }
}
