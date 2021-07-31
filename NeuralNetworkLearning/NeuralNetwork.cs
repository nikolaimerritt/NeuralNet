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
    using TrainingPairs = List<(Vector<double>, Vector<double>)>;

	public class NeuralNetwork
    {
        private Parameter _param;
        private DifferentiableFunction[] _activators;
        private static Random _rng = new();

        public int LayerCount
        {
            get => _param.LayerCount;
        }

        public NeuralNetwork(Parameter param, DifferentiableFunction[] activators)
        {
            if (activators.Length != param.LayerCount - 1)
                throw new ArgumentException($"Expected {param.LayerCount - 1} activators, but received {activators.Length}");

            _param = param;
            _activators = activators;
        }

        public NeuralNetwork(int[] layerSizes, DifferentiableFunction[] activators)
            : this(Parameter.StdUniform(layerSizes), activators) { }

        
        private Parameter AverageGradient(Parameter param, TrainingPairs trainingPairs) // assuming that trainingPairs.Count() is not extremely large
        {
            var gradients = trainingPairs
                .Select(pair => param.CostGrad(input: pair.Item1, desiredOutput: pair.Item2, _activators));

            return gradients.Aggregate((left, right) => left + right) / gradients.Count();
        }

        private Parameter VanillaGradDescentStep(Parameter param, TrainingPairs trainingPairs, double learningRate)
        {
            return learningRate * -AverageGradient(param, trainingPairs);
        }

        private Parameter NesterovGradDescentStep(Parameter param, TrainingPairs trainingPairs, Parameter prevParam, Parameter nesterovParam, double learningRate, int stepNumber, out Parameter nextNesterovParam)
        {
            Parameter nextParam = nesterovParam + learningRate * -AverageGradient(nesterovParam, trainingPairs);
            nextNesterovParam = nextParam + (stepNumber / (stepNumber + 3)) * (nextParam - param);

            return nextParam - param;
        }

        private Parameter AdamGradDescentStep(Parameter param, TrainingPairs trainingPairs, Parameter momentum, Parameter variance, double learningRate, double momentDecay, double varianceDecay, int stepNumber)
        {
            Parameter grad = AverageGradient(param, trainingPairs);
            momentum = momentDecay * momentum + (1 - momentDecay) * grad;
            momentum /= (1 - Math.Pow(momentDecay, stepNumber));

            variance = varianceDecay * variance + (1 - varianceDecay) * grad.Pow(2);
            variance /= (1 - Math.Pow(varianceDecay, stepNumber));

            return -learningRate * momentum / variance.Pow(0.5).Add(1e-8);
        }

        public void VanillaGradientDescent(TrainingPairs trainingPairs, double learningRate = 1e-3, int batchSize = 256)
        {
            for (int batchIdx = 0; batchIdx < trainingPairs.Count; batchIdx += batchSize)
            {
                var trainingBatch = trainingPairs.GetRange(batchIdx, Math.Min(batchSize, trainingPairs.Count - batchIdx));
                _param += VanillaGradDescentStep(_param, trainingBatch, learningRate);
            }
        }

        public void NesterovGradientDescent(TrainingPairs trainingPairs, double learningRate = 1e-3, int batchSize = 256)
        {
            Parameter nesterovParam = _param;
            for (int batchIdx = 0; batchIdx < trainingPairs.Count; batchIdx += batchSize)
            {
                var trainingBatch = trainingPairs.GetRange(batchIdx, Math.Min(batchSize, trainingPairs.Count - batchIdx));
                _param += NesterovGradDescentStep(_param, trainingBatch, _param, nesterovParam, learningRate, batchIdx, out nesterovParam);
            }
        }

        public void AdamGradientDescent(TrainingPairs trainingPairs, double learningRate = 1e-3, double momentDecay = 0.9, double varianceDecay = 0.99, int batchSize = 256)
        {
            Parameter momentum = Parameter.Zero(_param);
            Parameter variance = Parameter.Zero(_param);

            for (int batchIdx = 0; batchIdx < trainingPairs.Count; batchIdx += batchSize)
            {
                var trainingBatch = trainingPairs.GetRange(batchIdx, Math.Min(batchSize, trainingPairs.Count - batchIdx));
                _param += AdamGradDescentStep(_param, trainingBatch, momentum, variance, learningRate, momentDecay, varianceDecay, batchIdx);
            }
        }

        private static void Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = _rng.Next(n + 1); // 0 <= k <= n
                T putAtPosN = list[k];
                list[k] = list[n];
                list[n] = putAtPosN;
            }
        }
    }
}
