using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using NeuralNetLearning.Maths;

namespace NeuralNetLearning
{
    public class StochasticGradientDescender : GradientDescender
    {
        private readonly double _learningRate;

        public StochasticGradientDescender(double learningRate = 0.001)
            => _learningRate = learningRate;

        public string[] HyperParametersToLines()
            => new string[] { _learningRate.ToString() };

        public override Parameter GradientDescentStep(Parameter gradient)
            => _learningRate * -gradient;

        public static StochasticGradientDescender Read(string directoryPath)
        {
            double learningRate = HyperParamEncoder.Decode($"{directoryPath}/{hyperParamsFile}", "learning rate");
            return new StochasticGradientDescender(learningRate);
        }

        public override void WriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            HyperParamEncoder.EncodeToFile(
                this.GetType().Name,
                $"{directoryPath}/{hyperParamsFile}",
                ("learning rate", _learningRate)
            );
        }
    }
}
