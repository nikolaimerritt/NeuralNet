using NeuralNetLearning.Serialization;
using System;
using System.IO;

namespace NeuralNetLearning.Maths.GradientDescenders
{
    public class StochasticGradientDescender : GradientDescender
    {
        [SerializableHyperParameter("learning rate")]
        private readonly double _learningRate = 0.001;

        public StochasticGradientDescender(double learningRate)
            => _learningRate = learningRate;

        public StochasticGradientDescender()
        { }

        public string[] HyperParametersToLines()
            => new string[] { _learningRate.ToString() };

        internal override Parameter GradientDescentStep(Parameter gradient)
        {
            if (!gradient.IsFinite())
                throw new ArithmeticException($"Found non-finite gradient");
            return -_learningRate * gradient;
        }

        /* public static StochasticGradientDescender Read(string directoryPath)
        {
            double learningRate = HyperParamEncoder.Decode($"{directoryPath}/{_simpleHyperParamsFile}", "learning rate");
            return new StochasticGradientDescender(learningRate);
        } */

        /*public override void WriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            HyperParamEncoder.EncodeToFile(
                this.GetType().Name,
                $"{directoryPath}/{_simpleHyperParamsFile}",
                ("learning rate", _learningRate)
            );
        } */
    }
}
