using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using Maths;

namespace NeuralNetLearning
{
	public class StochasticNeuralNet : NeuralNet
	{
		private double learningRate;

		public StochasticNeuralNet(Parameter param, Activation[] activators, double learningRate = 1e-3)
			: base(param, activators)
		{
			this.learningRate = learningRate;
		}

		public StochasticNeuralNet(int[] layerSizes, Activation[] activators, double learningRate = 1e-3)
			: base(layerSizes, activators)
        {
			this.learningRate = learningRate;
        }

        protected override Parameter GradientDescentStep(Parameter grad)
        {
			return -learningRate * grad;
        }

		protected override string[] HyperParamsToLines()
			=> new string[] { learningRate.ToString() };

        protected override void SetHyperParamsFromFileContents(string[] lines)
        {
			double[] values = lines
				.Select(Double.Parse)
				.ToArray();

			if (values.Length != 1)
				throw new ArgumentException($"{values.Length} hyper-parameter values were supplied, but only 1 was expected");

			this.learningRate = values[0];
        }
    }
}
