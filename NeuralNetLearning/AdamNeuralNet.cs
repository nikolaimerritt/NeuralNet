using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using Maths;

namespace NeuralNetLearning
{
	public class AdamNeuralNet : NeuralNet
	{
		private double learningRate;
		private double momentumDecay;
		private double varianceDecay;

		private Parameter momentum;
		private Parameter variance;

		private int stepNumber = 0;

		public AdamNeuralNet(Parameter param, Activation[] activators, double learningRate = 0.001, double momentumDecay = 0.9, double varianceDecay = 0.999)
			: base(param, activators)
        {
			this.learningRate = learningRate;
			this.momentumDecay = momentumDecay;
			this.varianceDecay = varianceDecay;

			momentum = Parameter.Zero(param);
			variance = Parameter.Zero(param);
        }

		public AdamNeuralNet(int[] layerSizes, Activation[] activators, double learningRate = 1e-3, double momentumDecay = 0.9, double varianceDecay = 0.999)
			: base(layerSizes, activators)
        {
			this.learningRate = learningRate;
			this.momentumDecay = momentumDecay;
			this.varianceDecay = varianceDecay;

			momentum = Parameter.Zero(param);
			variance = Parameter.Zero(param);
		}

		public AdamNeuralNet(string directoryPath)
			: base(directoryPath)
        {
			double[] valuesRead = File.ReadAllLines($"{directoryPath}/{hyperParamsFileName}")
				.Select(Double.Parse)
				.ToArray();
        }

        protected override Parameter GradientDescentStep(Parameter grad)
        {
			if (stepNumber == 0)
            {
				momentum = grad;
				variance = grad.Pow(2);
            }
			else
            {
				momentum = momentumDecay * momentum + (1 - momentumDecay) * grad;
				variance = varianceDecay * variance + (1 - varianceDecay) * grad.Pow(2);
			}
			stepNumber++;

			var x = (momentum - grad).SquaredNorm();
			var y = (variance - grad.Pow(2)).SquaredNorm();

			var z = -learningRate * momentum / variance.Pow(0.5).Add(1e-8);
			return z;
		}

		protected override string[] HyperParamsToLines()
			=> new string[]
			{
				learningRate.ToString(),
				momentumDecay.ToString(),
				varianceDecay.ToString()
			};

        protected override void SetHyperParamsFromFileContents(string[] lines)
        {
			double[] values = lines
				.Select(Double.Parse)
				.ToArray();

			if (values.Length != 3)
				throw new ArgumentException($"{values.Length} hyper-parameter values were supplied, but only 3 were expected");

			learningRate = values[0];
			momentumDecay = values[1];
			varianceDecay = values[2];
        }
    }
}
