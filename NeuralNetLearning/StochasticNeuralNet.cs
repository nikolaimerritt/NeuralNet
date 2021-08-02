using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;

namespace NeuralNetLearning
{
	using Maths;
	public class StochasticNeuralNet : NeuralNet
	{
		private double _learningRate;

		public StochasticNeuralNet(IList<NeuralLayer> layerConfigs, double learningRate = 1e-3)
			: base(layerConfigs)
		{
			_learningRate = learningRate;
		}

        protected override Parameter GradientDescentStep(Parameter grad)
        {
			return -_learningRate * grad;
        }

		protected override string[] HyperParamsToLines()
			=> new string[] { _learningRate.ToString() };

        protected override void SetHyperParamsFromFileContents(string[] lines)
        {
			double[] values = lines
				.Select(Double.Parse)
				.ToArray();

			if (values.Length != 1)
				throw new ArgumentException($"{values.Length} hyper-parameter values were supplied, but only 1 was expected");

			_learningRate = values[0];
        }
    }
}
