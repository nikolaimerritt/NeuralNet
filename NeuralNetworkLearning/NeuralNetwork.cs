using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;

namespace NeuralNetLearning
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;

	public class NeuralNetwork
	{
		private readonly NeuralLayer[] _layers;
        private readonly NeuralLayerConfig processFinalLayer = NeuralLayerConfig.SigmoidConfig;
        public int LayerCount
        {
            get => _layers.Length;
        }

        #region Constructors
        public NeuralNetwork(NeuralLayer[] layers)
		{
			_layers = layers;
		}

        public NeuralNetwork(params int[] layerSizes)
        {
            List<NeuralLayer> layers = new();
            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                int inputDim = layerSizes[i];
                int outputDim = layerSizes[i + 1];
                NeuralLayerConfig config = DefaultConfig(i, layerSizes.Length - 1);

                layers.Add(new(inputDim, outputDim, config));
            }
            _layers = layers.ToArray();
        }

        public NeuralNetwork(string directoryPath)
        {
            List<string> weightPaths = Directory.GetFiles(directoryPath, "weight *.csv").ToList();
            weightPaths.Sort();
            List<string> biasPaths = Directory.GetFiles(directoryPath, "bias *.csv").ToList();
            biasPaths.Sort();

            List<NeuralLayer> layers = new();
            for (int i = 0; i < weightPaths.Count; i++)
            {
                NeuralLayerConfig config = DefaultConfig(layerPosition: i, maxLayerPosition: weightPaths.Count - 1);
                layers.Add(new(weightPaths[i], biasPaths[i], config));
            }
            _layers = layers.ToArray();
        }
        #endregion Constructors

        private static NeuralLayerConfig DefaultConfig(int layerPosition, int maxLayerPosition)
        {
            if (layerPosition == 0)
                return NeuralLayerConfig.IdentityConfig;

            if (layerPosition == maxLayerPosition)
                return NeuralLayerConfig.SigmoidConfig;

            else
                return NeuralLayerConfig.ReluConfig;
        }

        public void WriteToDirectory(string directoryPath)
        {
            for (int i = 0; i < LayerCount; i++)
            {
                string weightPath = $"{directoryPath}/weight {i+1}.csv";
                string biasPath = $"{directoryPath}/bias {i+1}.csv";
                _layers[i].Write(weightPath, biasPath);
            }
        }

        private Vector[] LayerValues(Vector input)
        {
            List<Vector> layerValues = new() { input };
            foreach (NeuralLayer layer in _layers)
            {
                layerValues.Add(layer.LayerValue(layerValues.Last()));
            }
            return layerValues.ToArray();
        }

        public Vector Output(Vector input)
            => processFinalLayer.Activator(LayerValues(input).Last());

        public void StochasticGradientDescent(Vector input, Vector desiredOutput, Func<Vector, Vector, Vector> costDeriv, double learningRate)
        {
            Vector[] layerValues = LayerValues(input);
            Vector output = processFinalLayer.Activator(layerValues.Last());

            Vector costGradWrtLayer = costDeriv(output, desiredOutput);
            for (int i = LayerCount - 1; i >= 0; i--)
            {
                Vector layerBehind = layerValues[i];
                _layers[i].GradientDescent(costGradWrtLayer, layerBehind, learningRate, out costGradWrtLayer);
            }
        }
    }
}
