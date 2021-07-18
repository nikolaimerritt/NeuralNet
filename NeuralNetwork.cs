using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;

namespace MachineLearning
{
	public class NeuralNetwork
	{
		private readonly NeuralLayer[] _layers;
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
            List<string> weightPaths = Directory.GetFiles(directoryPath, "weight_*.csv").ToList();
            weightPaths.Sort();
            List<string> biasPaths = Directory.GetFiles(directoryPath, "bias_*.csv").ToList();
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
                string weightPath = $"{directoryPath}/weight_{i}.csv";
                string biasPath = $"{directoryPath}/bias_{i}.csv";
                _layers[i].Write(weightPath, biasPath);
            }
        }

        public Vector<double> Output(Vector<double> input)
        {
            Vector<double> output = input;
            foreach (NeuralLayer layer in _layers)
            {
                output = layer.LayerValue(output);
            }
            return output;
        }
    }
}
