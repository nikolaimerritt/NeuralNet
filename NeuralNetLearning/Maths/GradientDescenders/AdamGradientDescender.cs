using NeuralNetLearning.Serialization;
using System;
using System.Linq;
using System.IO;


namespace NeuralNetLearning.Maths.GradientDescenders
{
    public class AdamGradientDescender : GradientDescender
    {
		[SerializableHyperParameter("learning rate")]
		private readonly double _learningRate = 0.001;

		[SerializableHyperParameter("momentum decay")]
		private readonly double _momentumDecay = 0.9;
		
		[SerializableHyperParameter("variance decay")]
		private readonly double _varianceDecay = 0.999;

		[SerializableHyperParameter("step")]
		private int _stepNum = 1;

		[SerializableHyperParameter("momentum")]
		private Parameter _momentum = null;

		[SerializableHyperParameter("variance")]
		private Parameter _variance = null;

		private static readonly double _preventDivByZero = 1e-10;

        public AdamGradientDescender(double learningRate, double momentumDecay, double varianceDecay, int step, Parameter momentum, Parameter variance)
        {
			_learningRate = learningRate;
			_momentumDecay = momentumDecay;
			_varianceDecay = varianceDecay;
			_stepNum = step;

			if (momentum != null && variance != null)
            {
				if (!momentum.LayerSizes.SequenceEqual(variance.LayerSizes))
					throw new ArgumentException($"Expected momentum and variance to have the same layer count and entries count.");
			}
			
			_momentum = momentum;
			_variance = variance;
        }

		public AdamGradientDescender()
		{ 
		
		}


		internal override Parameter GradientDescentStep(Parameter gradient)
		{
			_stepNum++;

			if (_momentum == null)
				_momentum = ParameterFactory.Zero(gradient.LayerSizes);

			if (_variance == null)
				_variance = ParameterFactory.Zero(gradient.LayerSizes);

			_momentum = _momentumDecay * _momentum + (1 - _momentumDecay) * gradient;
			_variance = _varianceDecay * _variance + (1 - _varianceDecay) * gradient.Pow(2);

			Parameter correctedMomentum = _momentum / (1 - Math.Pow(_momentumDecay, _stepNum));
			Parameter correctedVariance = _variance / (1 - Math.Pow(_varianceDecay, _stepNum));
			
			Parameter step = -_learningRate * correctedMomentum / (correctedVariance.Pow(0.5) + _preventDivByZero);
			return step;
		}

		/*
		public override void WriteToDirectory(string directoryPath)
        {
			if (!Directory.Exists(directoryPath))
				Directory.CreateDirectory(directoryPath);

			HyperParamEncoder.EncodeToFile(
				this.GetType().Name, 
				$"{directoryPath}/{_simpleHyperParamsFile}",
				("learning rate", _learningRate),
				("momentum decay", _momentumDecay),
				("variance decay", _varianceDecay),
				("step", _stepNum)
			);
			_momentum?.WriteToDirectory($"{directoryPath}/{momentumFolder}");
			_variance?.WriteToDirectory($"{directoryPath}/{varianceFolder}");
        }

        public static AdamGradientDescender Read(string directoryPath)
        {
			string filepath = $"{directoryPath}/{_simpleHyperParamsFile}";
			double learningRate = HyperParamEncoder.Decode(filepath, "learning rate");
			double momentumDecay = HyperParamEncoder.Decode(filepath, "momentum decay");
			double varianceDecay = HyperParamEncoder.Decode(filepath, "variance decay");
			int step = (int)HyperParamEncoder.Decode(filepath, "step");

			Parameter momentum = File.Exists($"{directoryPath}/{momentumFolder}") ?
				ParameterFactory.ReadFromDirectory($"{directoryPath}/{momentumFolder}")
				: null;

			Parameter variance = File.Exists($"{directoryPath}/{varianceFolder}") ?
				ParameterFactory.ReadFromDirectory($"{directoryPath}/{varianceFolder}")
				: null;

			return new AdamGradientDescender(learningRate, momentumDecay, varianceDecay, step, momentum, variance);
        } */
    }
}
