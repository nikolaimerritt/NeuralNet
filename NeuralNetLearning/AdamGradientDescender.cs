using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using NeuralNetLearning.Maths;


namespace NeuralNetLearning
{
    public class AdamGradientDescender : GradientDescender
    {
		private readonly double _learningRate;
		private readonly double _momentumDecay;
		private readonly double _varianceDecay;
		private int _step;

		private Parameter _momentum;
		private Parameter _variance;

		private static readonly double _preventDivByZero = 1e-10;
		private static readonly string momentumFolder = "momentum";
		private static readonly string varianceFolder = "variance";

        private AdamGradientDescender(double learningRate, double momentumDecay, double varianceDecay, int step, Parameter momentum, Parameter variance)
        {
			_learningRate = learningRate;
			_momentumDecay = momentumDecay;
			_varianceDecay = varianceDecay;
			_step = step;

			if (momentum != null && variance != null)
            {
				if (momentum.EntriesCount != variance.EntriesCount || momentum.LayerCount != variance.LayerCount)
					throw new ArgumentException($"Expected momentum and variance to have the same layer count and entries count.");
			}
			
			_momentum = momentum;
			_variance = variance;
        }

		public AdamGradientDescender(double learningRate = 0.001, double momentumDecay = 0.9, double varianceDecay = 0.999)
			: this(learningRate, momentumDecay, varianceDecay, step: 1, momentum: null, variance: null)
		{ }


		public override Parameter GradientDescentStep(Parameter gradient)
		{
			_step++;

			if (_momentum == null)
				_momentum = 0 * gradient;

			if (_variance == null)
				_variance = 0 * gradient;

			_momentum = _momentumDecay * _momentum + (1 - _momentumDecay) * gradient;
			_variance = _varianceDecay * _variance + (1 - _varianceDecay) * gradient.Pow(2);

			Parameter correctedMomentum = _momentum / (1 - Math.Pow(_momentumDecay, _step));
			Parameter correctedVariance = _variance / (1 - Math.Pow(_varianceDecay, _step));
			Parameter step = -_learningRate * correctedMomentum / correctedVariance.Pow(0.5).Add(_preventDivByZero);
			if (!step.IsFinite())
				throw new ArithmeticException($"Found non-finite value");
			return step;
		}

		public override void WriteToDirectory(string directoryPath)
        {
			if (!Directory.Exists(directoryPath))
				Directory.CreateDirectory(directoryPath);

			HyperParamEncoder.EncodeToFile(
				this.GetType().Name, 
				$"{directoryPath}/{hyperParamsFile}",
				("learning rate", _learningRate),
				("momentum decay", _momentumDecay),
				("variance decay", _varianceDecay),
				("step", _step)
			);
			_momentum?.WriteToDirectory($"{directoryPath}/{momentumFolder}");
			_variance?.WriteToDirectory($"{directoryPath}/{varianceFolder}");
        }

        public static AdamGradientDescender Read(string directoryPath)
        {
			string filepath = $"{directoryPath}/{hyperParamsFile}";
			double learningRate = HyperParamEncoder.Decode(filepath, "learning rate");
			double momentumDecay = HyperParamEncoder.Decode(filepath, "momentum decay");
			double varianceDecay = HyperParamEncoder.Decode(filepath, "variance decay");
			int step = (int)HyperParamEncoder.Decode(filepath, "step");

			Parameter momentum = File.Exists($"{directoryPath}/{momentumFolder}") ?
				Parameter.ReadFromDirectory($"{directoryPath}/{momentumFolder}")
				: null;

			Parameter variance = File.Exists($"{directoryPath}/{varianceFolder}") ?
				Parameter.ReadFromDirectory($"{directoryPath}/{varianceFolder}")
				: null;

			return new AdamGradientDescender(learningRate, momentumDecay, varianceDecay, step, momentum, variance);
        }
    }
}
