using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using Maths;
/*
namespace NeuralNetLearning
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;

	public class NeuralNetwork
	{
        private readonly Parameters _parameters;
        private readonly Activator[] _activators;
        public static readonly string DefaultDirectory = "../../../NeuralNetworkLearning/layers";
        public int LayerCount
        {
            get => _parameters.LayerCount;
        }

        public NeuralNetwork(Parameters parameters, IEnumerable<Activator> activators)
		{
            _parameters = parameters;
            _activators = activators.ToArray();
		}


        public NeuralNetwork(Parameters parameters)
            : this(parameters, DefaultActivators(parameters.LayerCount)) { }


        public NeuralNetwork(params int[] layerSizes)
            : this(new Parameters(layerSizes), DefaultActivators(layerSizes.Length)) { }


        private static IEnumerable<Activator> DefaultActivators(int layerCount)
            => Enumerable
            .Range(0, layerCount)
            .Select(i => i == 0 ? Activator.Identity : Activator.Relu);


        public static NeuralNetwork ReadFromDirectory(string directoryPath)
            => new (Parameters.ReadFromDirectory(directoryPath));
        

        public static NeuralNetwork ReadFromDirectory()
            => ReadFromDirectory(DefaultDirectory);



        public void WriteToDirectory(string directoryPath)
           => _parameters.WriteToDirectory(directoryPath);

        public void WriteToDirectory()
            => WriteToDirectory(DefaultDirectory);


        public double Cost(Vector input, Vector desiredOutput)
            => VectorFunctions.MSE(Output(input), desiredOutput);


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
*/