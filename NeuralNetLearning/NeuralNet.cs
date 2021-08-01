using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Threading.Tasks;
using Maths;

namespace NeuralNetLearning
{
    using Matrix = Matrix<double>;
    using Vector = Vector<double>;
    using TrainingPairs = List<(Vector<double>, Vector<double>)>;

	public abstract class NeuralNet
    {
        protected Parameter param;
        protected Activation[] activators;
        private readonly static Random rng = new();

        private static readonly string activatorsFileName = "activators.txt";
        protected static readonly string hyperParamsFileName = "hyperparameters.txt";

        public int LayerCount
        {
            get => param.LayerCount;
        }

        public int[] LayerSizes
        {
            get => param.LayerSizes;
        }

        protected NeuralNet(Parameter param, Activation[] activators)
        {
            if (activators.Length != param.LayerCount)
                throw new ArgumentException($"Expected {param.LayerCount} activators, but received {activators.Length}");

            this.param = param;
            this.activators = activators;
        }

        protected NeuralNet(int[] layerSizes, Activation[] activators)
            : this(Parameter.StdUniform(layerSizes), activators) 
        { }

        protected NeuralNet(string directoryPath)
            : this(Parameter.Read(directoryPath), ReadActivators(directoryPath))
        {
            SetHyperParamsFromFileContents(File.ReadAllLines($"{directoryPath}/{hyperParamsFileName}"));
        }

        protected abstract void SetHyperParamsFromFileContents(string[] lines);

        private static Activation[] ReadActivators(string directoryPath)
            => File
                .ReadAllLines($"{directoryPath}/{activatorsFileName}")
                .Select(name => (Activation) Enum.Parse(typeof(Activation), name))
                .ToArray();

        protected void WriteParameter(string directoryPath)
            => param.WriteToDirectory(directoryPath);

        protected abstract string[] HyperParamsToLines();

        public void WriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            param.WriteToDirectory(directoryPath);
            File.WriteAllLines($"{directoryPath}/{activatorsFileName}", activators.Select(a => a.ToString()));
            File.WriteAllLines($"{directoryPath}/{hyperParamsFileName}", HyperParamsToLines());
        }

        protected Parameter AverageGradient(Parameter param, TrainingPairs trainingPairs) // assuming that trainingPairs.Count() is not extremely large
        {
            var gradients = trainingPairs
                .Select(pair => param.CostGrad(input: pair.Item1, desiredOutput: pair.Item2, activators));

            return gradients.Aggregate((left, right) => left + right) / gradients.Count();
        }

        protected abstract Parameter GradientDescentStep(Parameter grad);

        public void GradientDescent(TrainingPairs trainingPairs, int batchSize = 256)
        {
            for (int batchIdx = 0; batchIdx < trainingPairs.Count; batchIdx += batchSize)
            {
                var trainingBatch = trainingPairs.GetRange(batchIdx, Math.Min(batchSize, trainingPairs.Count - batchIdx));
                Console.WriteLine($"Avg training cost: {AverageCost(trainingBatch)}");
                param += GradientDescentStep(AverageGradient(param, trainingBatch));
            }
        }

        public Vector Output(Vector input)
            => param.Output(input, activators);

        public double AverageCost(TrainingPairs trainingPairs)
            => trainingPairs
            .Select(pair => param.Cost(pair.Item1, pair.Item2, activators))
            .Average();


        private static void Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1); // 0 <= k <= n
                T putAtPosN = list[k];
                list[k] = list[n];
                list[n] = putAtPosN;
            }
        }
    }
}
