using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Threading.Tasks;
using NeuralNetLearning.Maths;

namespace NeuralNetLearning
{
    using Vector = Vector<double>;
    using VectorPairs = List<(Vector<double>, Vector<double>)>;

	public abstract class NeuralNet
    {
        protected Parameter _param;
        protected Activation[] _activators;
        private readonly static Random _rng = new();

        private static readonly string _activatorsFile = "activators.txt";
        protected static readonly string _hyperParamsFile = "hyperparameters.txt";

        public int LayerCount
        {
            get => _param.LayerCount;
        }

        public int[] LayerSizes
        {
            get => _param.LayerSizes;
        }

        
        private NeuralNet(Parameter param, Activation[] activators)
        {
            if (activators.Length != param.LayerCount)
                throw new ArgumentException($"Expected {param.LayerCount} activators, but received {activators.Length}");

            this._param = param;
            this._activators = activators;
        }

        protected static int[] LayerSizesFromConfigs(IList<NeuralLayer> layerConfigs)
            => layerConfigs
                .Select(l => l.LayerSize)
                .ToArray();

        private static Activation[] ActivationsFromConfigs(IList<NeuralLayer> layerConfigs)
        {
            if (!(layerConfigs.First() is InputLayer))
                throw new ArgumentException($"Expected the first layer to be of type {typeof(InputLayer)}");

            List<Activation> activators = new(layerConfigs.Count - 1);
            for (int i = 1; i < layerConfigs.Count - 1; i++)
            {
                if (!(layerConfigs[i] is HiddenLayer))
                    throw new ArgumentException($"Expected layer {i} to be of type {typeof(HiddenLayer)}");

                activators.Add((layerConfigs[i] as HiddenLayer).Activator);
            }

            if (!(layerConfigs.Last() is OutputLayer))
                throw new ArgumentException($"Expected the last layer to be of type {typeof(OutputLayer)}");

            activators.Add((layerConfigs.Last() as OutputLayer).Activator);

            return activators.ToArray();
        }

        protected NeuralNet(IList<NeuralLayer> layerConfigs)
            : this(Parameter.StdUniform(LayerSizesFromConfigs(layerConfigs)), ActivationsFromConfigs(layerConfigs))
        { }

        protected NeuralNet(string directoryPath)
            : this(Parameter.Read(directoryPath), ReadActivators(directoryPath))
        {
            SetHyperParamsFromFileContents(File.ReadAllLines($"{directoryPath}/{_hyperParamsFile}"));
        }

        protected abstract void SetHyperParamsFromFileContents(string[] lines);

        private static Activation[] ReadActivators(string directoryPath)
            => File
                .ReadAllLines($"{directoryPath}/{_activatorsFile}")
                .Select(name => (Activation) Enum.Parse(typeof(Activation), name))
                .ToArray();

        protected void WriteParameter(string directoryPath)
            => _param.WriteToDirectory(directoryPath);

        protected abstract string[] HyperParamsToLines();

        public void WriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            _param.WriteToDirectory(directoryPath);
            File.WriteAllLines($"{directoryPath}/{_activatorsFile}", _activators.Select(a => a.ToString()));
            File.WriteAllLines($"{directoryPath}/{_hyperParamsFile}", HyperParamsToLines());
        }

        protected Parameter AverageGradient(Parameter param, VectorPairs trainingPairs) // assuming that trainingPairs.Count() is not extremely large
        {
            var gradients = trainingPairs
                .Select(pair => param.CostGrad(input: pair.Item1, desiredOutput: pair.Item2, _activators));

            return gradients.Aggregate((left, right) => left + right) / gradients.Count();
        }

        protected abstract Parameter GradientDescentStep(Parameter grad);


        public void GradientDescent(VectorPairs trainingPairs, int batchSize)
        {
            for (int batchIdx = 0; batchIdx < trainingPairs.Count; batchIdx += batchSize)
            {
                VectorPairs trainingBatch = trainingPairs
                    .GetRange(batchIdx, Math.Min(batchSize, trainingPairs.Count - batchIdx));

                Console.WriteLine($"Avg training cost: {AverageCost(trainingBatch)}");
                _param += GradientDescentStep(AverageGradient(_param, trainingBatch));
            }
        }

        protected Vector Output(Vector input)
            => _param.Output(input, _activators);

        public double AverageCost(VectorPairs trainingPairs)
            => trainingPairs
            .Select(pair => _param.Cost(pair.Item1, pair.Item2, _activators))
            .Average();
    }
}
