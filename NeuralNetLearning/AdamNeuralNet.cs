using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using NeuralNetLearning.Maths;

namespace NeuralNetLearning
{
	public class AdamNeuralNet : NeuralNet
	{
		private double _learningRate;
		private double _momentumDecay;
		private double _varianceDecay;

		private Parameter _momentum;
		private Parameter _variance;

		private int _stepNumber = 0;

		public AdamNeuralNet(IList<NeuralLayer> layerConfigs, double learningRate = 0.001, double momentumDecay = 0.9, double varianceDecay = 0.999)
			: base(layerConfigs)
        {
			_learningRate = learningRate;
			_momentumDecay = momentumDecay;
			_varianceDecay = varianceDecay;

			_momentum = Parameter.Zero(LayerSizesFromConfigs(layerConfigs));
			_variance = Parameter.Zero(LayerSizesFromConfigs(layerConfigs));
        }

		public AdamNeuralNet(string directoryPath)
			: base(directoryPath)
        {
			double[] valuesRead = File.ReadAllLines($"{directoryPath}/{_hyperParamsFile}")
				.Select(Double.Parse)
				.ToArray();
        }

        protected override Parameter GradientDescentStep(Parameter grad)
        {
			if (_stepNumber == 0)
            {
				_momentum = grad;
				_variance = grad.Pow(2);
            }
			else
            {
				_momentum = _momentumDecay * _momentum + (1 - _momentumDecay) * grad;
				_variance = _varianceDecay * _variance + (1 - _varianceDecay) * grad.Pow(2);
			}
			_stepNumber++;

			return -_learningRate * _momentum / _variance.Pow(0.5).Add(1e-8);
		}

		protected override string[] HyperParamsToLines()
			=> new string[]
			{
				_learningRate.ToString(),
				_momentumDecay.ToString(),
				_varianceDecay.ToString()
			};

        protected override void SetHyperParamsFromFileContents(string[] lines)
        {
			double[] values = lines
				.Select(Double.Parse)
				.ToArray();

			if (values.Length != 3)
				throw new ArgumentException($"{values.Length} hyper-parameter values were supplied, but only 3 were expected");

			_learningRate = values[0];
			_momentumDecay = values[1];
			_varianceDecay = values[2];
        }
    }
}
