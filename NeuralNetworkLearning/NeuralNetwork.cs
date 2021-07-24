using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using Maths;

namespace NeuralNetLearning
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;

	public class NeuralNetwork
	{
		public readonly NeuralLayer[] _layers;
        private readonly NeuralLayerConfig processFinalLayer = NeuralLayerConfig.ReluConfig;
        public static readonly string DefaultDirectory = "../../../NeuralNetworkLearning/layers";
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
                NeuralLayerConfig config = NeuralLayerConfig.ReluConfig;

                layers.Add(new(inputDim, outputDim, config));
            }
            _layers = layers.ToArray();
        }

        public static NeuralNetwork ReadFromDirectory(string directoryPath)
        {
            List<string> weightPaths = Directory.GetFiles(directoryPath, "weight *.csv").ToList();
            weightPaths.Sort();
            List<string> biasPaths = Directory.GetFiles(directoryPath, "bias *.csv").ToList();
            biasPaths.Sort();

            List<NeuralLayer> layers = new();
            for (int i = 0; i < weightPaths.Count; i++)
            {
                NeuralLayerConfig config = NeuralLayerConfig.ReluConfig;
                layers.Add(new(weightPaths[i], biasPaths[i], config));
            }
            return new NeuralNetwork(layers.ToArray());
        }

        public static NeuralNetwork ReadFromDirectory()
            => ReadFromDirectory(DefaultDirectory);
        #endregion Constructors



        public void WriteToDirectory(string directoryPath)
        {
            for (int i = 0; i < LayerCount; i++)
            {
                string weightPath = $"{directoryPath}/weight {i+1}.csv";
                string biasPath = $"{directoryPath}/bias {i+1}.csv";
                _layers[i].Write(weightPath, biasPath);
            }
        }

        public void WriteToDirectory()
            => WriteToDirectory(DefaultDirectory);

        private Vector[] LayerValues(Vector input)
        {
            List<Vector> layerValues = new() { input };
            foreach (NeuralLayer layer in _layers)
            {
                layerValues.Add(layer.LayerValue(layerValues.Last()));
            }
            return layerValues.ToArray();
        }


        public void SetWeight(int layerIdx, Matrix weight)
            => _layers[layerIdx].SetWeight(weight);
        public void SetBias(int layerIdx, Vector bias)
            => _layers[layerIdx].SetBias(bias);


        public Vector Output(Vector input)
            => processFinalLayer.Activator(LayerValues(input).Last());


        public double Cost(Vector input, Vector desiredOutput)
            => VectorFunctions.MSE(Output(input), desiredOutput);

        private Vector CostGradWrtFinalLayer(Vector finalLayer, Vector desiredOutput)
        {
            Vector output = processFinalLayer.Activator(finalLayer);
            Vector outerDeriv = VectorFunctions.MSEderiv(output, desiredOutput);
            Vector innerDeriv = processFinalLayer.ActivatorDeriv(finalLayer);
            return Vector.op_DotMultiply(outerDeriv, innerDeriv);
        }

        public (Matrix[], Vector[]) WeightsAndBiasesOfGradDescent(Vector input, Vector desiredOutput, double learningRate)
        {
            Vector[] layerValues = LayerValues(input);

            Vector costGradWrtLayer = CostGradWrtFinalLayer(layerValues.Last(), desiredOutput);
            List<Matrix> weightCostGrads = new();
            List<Vector> biasCostGrads = new();

            for (int i = LayerCount - 1; i >= 0; i--)
            {
                Vector layerBehind = layerValues[i];
                (Matrix weightGrad, Vector biasGrad) = _layers[i].GradientDescent(costGradWrtLayer, layerBehind, learningRate, out Vector costGradWrtLayerBehind);
                weightCostGrads.Add(weightGrad);
                biasCostGrads.Add(biasGrad);

                costGradWrtLayer = costGradWrtLayerBehind;
            }

            weightCostGrads.Reverse();
            biasCostGrads.Reverse();
            return (weightCostGrads.ToArray(), biasCostGrads.ToArray());
        }

        public NeuralNetwork DeepCopy()
        {
            NeuralLayer[] newLayers = _layers.Select(layer => layer.DeepCopy()).ToArray();
            return new(newLayers);
        }

        public NeuralNetwork DeepCopyWithModification(Matrix newWeight, int layerIdxToModify)
        {
            NeuralNetwork deepCopy = DeepCopy();
            deepCopy._layers[layerIdxToModify] = deepCopy._layers[layerIdxToModify].DeepCopyWithModification(newWeight);
            return deepCopy;
        }

        public NeuralNetwork DeepCopyWithModification(Vector newBias, int layerIdxToModify)
        {
            NeuralNetwork deepCopy = DeepCopy();
            deepCopy._layers[layerIdxToModify] = deepCopy._layers[layerIdxToModify].DeepCopyWithReplacement(newBias);
            return deepCopy;
        }
    }
}
