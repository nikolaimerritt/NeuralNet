using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NeuralNetLearning.Maths;

namespace NeuralNetLearning
{
    using TrainingPair = Tuple<Vector<double>, Vector<double>>;

    /// <summary>
    /// A neural network 
    /// </summary>
	public class NeuralNet
    {
        private readonly Parameter _param;
        private readonly CostFunction _cost;
        private readonly Activation[] _activators;
        private readonly GradientDescender _gradientDescender;
        private static readonly Random _rng = new();

        /// <summary>
        /// The total amount of layers, including the input layer, hidden layers and activation layer
        /// </summary>
        public int LayerCount
        {
            get => _param.LayerCount;
        }

        /// <summary>
        /// An array containing the size of each layer, including the input layer, hidden layers, and activation layer
        /// </summary>
        public int[] LayerSizes
        {
            get => _param.LayerSizes;
        }

        /// <summary>
        /// The default constructor. It is reccommended to use the NeuralNetFactory class to create NeuralNet objects for basic purposes.
        /// </summary>
        /// <param name="param"> The parameter object that the NeuralNet will use internally </param>
        /// <param name="activators"> 
            /// <para>The array of activators that the NeuralNet will use internally. Each activator is applied to the corresponding (non-output) layer stored in <paramref name="param"/>.  </para>
            /// <para>No activator is applied to the output layer <paramref name="param"/>. Thus, if <paramref name="param"/> has <c>n</c> layers, <paramref name="activators"/> must have <c>n-1</c> elements. </para>
        /// </param>

        public NeuralNet(Parameter param, Activation[] activators, GradientDescender gradientDescent, CostFunction cost)
        {
            if (activators.Length != param.LayerCount)
                throw new ArgumentException($"Expected {param.LayerCount} activators, but received {activators.Length}");

            _param = param;
            _cost = cost;
            _activators = activators;
            _gradientDescender = gradientDescent;
        }

        private Parameter AverageGradient(Parameter param, IReadOnlyList<TrainingPair> trainingPairs) // assuming that trainingPairs.Count() is not extremely large
        {
            if (!trainingPairs.Any())
                throw new ArgumentException("Could not find the average gradient of an empty list of training pairs");

            Parameter total = ParameterFactory.Zero(param.LayerSizes);
            foreach ((Vector<double> input, Vector<double> desiredOutput) in trainingPairs)
                total.InPlaceAdd(_param.CostGrad(input, desiredOutput, _activators, _cost));

            total.InPlaceDivide(trainingPairs.Count);
            return total;
        }

        private Parameter AverageGradientParallel(Parameter parameter, ArraySegment<TrainingPair> trainingPairs)
        {
            if (!trainingPairs.Any())
                throw new ArgumentException("Could not find the average gradient of an empty list of training pairs");

            var batches = Batches(trainingPairs, batchSize: Math.Max(1, trainingPairs.Count / Environment.ProcessorCount));
            int numThreads = batches.Count;
            
            Parameter[] threadAverages = new Parameter[numThreads];
            Parallel.For(0, numThreads, i => 
            {
                threadAverages[i] = ParameterFactory.Zero(parameter.LayerSizes);
                var batch = batches[i];
                foreach ((Vector<double> input, Vector<double> desiredOutput) in batch)
                {
                    Parameter grad = _param.CostGrad(input, desiredOutput, _activators, _cost);
                    threadAverages[i].InPlaceAdd(grad);
                }
                threadAverages[i].InPlaceDivide(batch.Count);
            });

            Parameter average = ParameterFactory.Zero(parameter.LayerSizes);
            foreach (Parameter threadAverage in threadAverages)
                average.InPlaceAdd(threadAverage);
            average.InPlaceDivide(threadAverages.Length);
            return average;
        }

        private static List<ArraySegment<TrainingPair>> Batches(ArraySegment<TrainingPair> trainingPairs, int batchSize)
        {
            int numBatches = (int)Math.Ceiling((double)trainingPairs.Count / batchSize);
            List<ArraySegment<TrainingPair>> batches = new(capacity: numBatches);
            for (int i = 0; i < numBatches; i++)
            {
                var batch = BoundedSlice(trainingPairs, start: i * batchSize, length: batchSize);
                if (batch.Any())
                    batches.Add(batch);
            }
            return batches;
        }

        /// <summary>
        /// Runs batch gradient descent on the training data stored in <paramref name="trainingPairs"/>. 
        /// </summary>
        /// <param name="trainingPairs">
            /// <para> A (finite) IEnumerable of training data. </para>
            /// <para> Each element of <paramref name="trainingPairs"/> is a tuple. The first element is the input to the NeuralNet, and the second element is the corresponding output the NeuralNet will learn to produce. </para>
        /// </param>
        /// <param name="batchSize"> The size of each batch that <paramref name="trainingPairs"/> will be split into. Recommended values of <paramref name="batchSize"/> range from 4 to 256. </param>
        /// <param name="numEpochs"> The number of times that batch gradient descent will be run on <paramref name="trainingPairs"/>. </param>
        /// <param name="batchUpdateInParallel"> If <paramref name="batchUpdateInParallel"/> is <c> true </c>, the average gradient corresponding to each batch is computed in parallel. Reccommended for medium to high values of <paramref name="batchSize"/>. </param>
        public void GradientDescent(IEnumerable<TrainingPair> trainingPairs, int batchSize = 256, int numEpochs = 100, bool batchUpdateInParallel = true)
        {
            ArraySegment<TrainingPair> trainingPairView = new(trainingPairs.ToArray());
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                Shuffle(trainingPairView);
                foreach (var batch in Batches(trainingPairView, batchSize))
                {
                    Parameter grad = batchUpdateInParallel ?
                        AverageGradientParallel(_param, batch) :
                        AverageGradient(_param, batch);
                    Parameter update = _gradientDescender.GradientDescentStep(grad);
                    _param.InPlaceAdd(update);
                }
            }
        }

        /// <summary>
        /// Writes the parameter values, activator type and hyperparameters, gradient descender type and hyperparameters, and cost type to the specified directory in a human-readable format.
        /// </summary>
        /// <param name="directoryPath"> The (relative or absolute) path of the directory to be written to. </param>
        public void WriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            _param.WriteToDirectory($"{directoryPath}/{NeuralNetFactory.ParamsFolder}");
            WriteActivatorsToDirectory($"{directoryPath}/{NeuralNetFactory.ActivatorsFolder}");
            _gradientDescender.WriteToDirectory($"{directoryPath}/{NeuralNetFactory.GradientDescenderFolder}");
            _cost.WriteToFile($"{directoryPath}/{NeuralNetFactory.CostFile}");
        }

        private void WriteActivatorsToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            for (int i = 0; i < _activators.Length; i++)
                _activators[i].WriteToFile($"{directoryPath}/activator {i + 1}.txt");
        }

        /// <summary>
        /// Calculates the output of the Neural Network for the vector <paramref name="input"/>.
        /// </summary>
        public Vector<double> Output(Vector<double> input)
            => _param.Output(input, _activators);

        /// <summary>
        /// Calculates the average of the error cost between the output the neural network generates and corresponding expected output.
        /// </summary>
        /// <param name="testingPairs"> The list of <c>(input, expected output)</c> testing pairs. The cost of the error between the Neural network's output to each input, and the corresponding expected output, is averaged. </param>
        /// <returns></returns>
        public double AverageCost(IEnumerable<TrainingPair> testingPairs)
            => testingPairs
            .Select(pair => _cost.Apply(Output(pair.Item1), pair.Item2))
            .Average();

        private static void Shuffle<T>(IList<T> list)
        {
            int swapIdx = list.Count;
            while (swapIdx > 1)
            {
                swapIdx--;
                int replaceIdx = _rng.Next(swapIdx + 1);
                T value = list[replaceIdx];
                list[replaceIdx] = list[swapIdx];
                list[swapIdx] = value;
            }
        }

        private static ArraySegment<T> BoundedSlice<T>(ArraySegment<T> segment, int start, int length)
            => segment.Slice(start, Math.Min(length, segment.Count - start));
    }
}