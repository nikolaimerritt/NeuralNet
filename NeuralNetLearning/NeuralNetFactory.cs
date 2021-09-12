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
    using Matrix = Matrix<double>;
    using static ParameterFactory;

    public static class NeuralNetFactory
    {
        /// <summary>
        /// The name of the file in which the CostFunction of a NeuralNet is recorded when writing a NeuralNet to a directory.
        /// </summary>
        public static readonly string CostFile = "cost.txt";
        /// <summary>
        /// The name of the folder in which the Parameter of a NeuralNet is recorded when writing a NeuralNet to a directory.
        /// </summary>
        public static readonly string ParamsFolder = "parameters";
        /// <summary>
        /// The name of the folder in which the Activators of a NeuralNet are recorded when writing a NeuralNet to a directory.
        /// </summary>
        public static readonly string ActivatorsFolder = "activators";
        /// <summary>
        /// The name of the folder in which the Gradient Descender of a NeuralNet is recorded when writing a NeuralNet to a directory.
        /// </summary>
        public static readonly string GradientDescenderFolder = "gradient-descender";

        /// <summary>
        /// Returns a NeuralNet initialised with random weights and biases that are optimised for the use of TanhSigmoid activators. Uses Xavier initialisation.
        /// </summary>
        /// <param name="layerConfigs"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet RandomOptimisedForTanhSigmoid(IList<NeuralLayerConfig> layerConfigs, GradientDescender gradientDescender, CostFunction cost)
        {
            Parameter param = XavierInit(LayerSizesFromConfigs(layerConfigs));
            Activation[] activators = ActivationsFromConfigs(layerConfigs);

            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        /// <summary>
        /// Returns a NeuralNet initialised with random weights and biases that are optimised for the use of Relu activators with low or zero leak. Uses Kaiming-He initialisation.
        /// </summary>
        /// <param name="layerConfigs"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet RandomOptimisedForRelu(IList<NeuralLayerConfig> layerConfigs, GradientDescender gradientDescender, CostFunction cost)
        {
            Parameter param = KaimingInit(LayerSizesFromConfigs(layerConfigs));
            Activation[] activations = ActivationsFromConfigs(layerConfigs);

            return new NeuralNet(param, activations, gradientDescender, cost);
        }

        /// <summary>
        /// Returns a NeuralNet initialised with random weights and biases that are optimised for learning the data set supplied in <paramref name="sampleTrainingSet"/>. Uses LSUV initialisation.
        /// </summary>
        /// <param name="layerConfigs"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="sampleTrainingSet"> The training set which the NeuralNet will be optimised to learn. Only the inputs are needed, so this method simply calls its overload where <paramref name="sampleTrainingSet"/> is a list of inputs. </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet RandomCustomisedForMiniBatch(IList<NeuralLayerConfig> layerConfigs, IEnumerable<(Vector input, Vector desiredOutput)> sampleTrainingSet, GradientDescender gradientDescender, CostFunction cost)
        {
            IList<Vector> miniBatchInputs = sampleTrainingSet.Select(pair => pair.input).ToList();
            return RandomCustomisedForMiniBatch(layerConfigs, miniBatchInputs, gradientDescender, cost);
        }

        /// <summary>
        /// Returns a NeuralNet initialised with random weights and biases that are optimised for learning the inputs supplied in <paramref name="sampleInputs"/>. Uses LSUV initialisation.
        /// </summary>
        /// <param name="layerConfigs"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="sampleInputs"> The inputs of the training set which the NeuralNet will be optimised to learn. </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet RandomCustomisedForMiniBatch(IList<NeuralLayerConfig> layerConfigs, IEnumerable<Vector> sampleInputs, GradientDescender gradientDescender, CostFunction cost)
        {
            Activation[] activators = ActivationsFromConfigs(layerConfigs);
            Parameter param = LSUVInit(LayerSizesFromConfigs(layerConfigs), activators, sampleInputs);
            
            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        /// <summary>
        /// Returns a NeuralNet read from the supplied directory. The NeuralNet must have been written in the same format as in the <c> NeuralNet.WriteToDirectory </c> function.
        /// </summary>
        /// <param name="directoryPath"> The (absolute or relative) path to which a Neural Net has been written. </param>
        /// <returns></returns>
        public static NeuralNet ReadFromDirectory(string directoryPath)
        {
            Parameter param = Parameter.ReadFromDirectory($"{directoryPath}/{ParamsFolder}");
            Activation[] activators = ReadActivationsFromDirectory($"{directoryPath}/{ActivatorsFolder}");
            GradientDescender gradientDescender = GradientDescender.ReadFromDirectory($"{directoryPath}/{GradientDescenderFolder}");
            CostFunction cost = CostFunction.ReadFromFile($"{directoryPath}/{CostFile}");

            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        private static Activation[] ReadActivationsFromDirectory(string directory)
        {
            if (!Directory.Exists(directory))
                throw new FileNotFoundException($"Could not find directory {directory}");

            List<string> activationFiles = Directory.GetFiles(directory).ToList();
            activationFiles.Sort();

            return activationFiles.Select(Activation.ReadFromFile).ToArray();
        }


        private static int[] LayerSizesFromConfigs(IList<NeuralLayerConfig> layerConfigs)
                => layerConfigs
                    .Select(l => l.LayerSize)
                    .ToArray();

        private static Activation[] ActivationsFromConfigs(IList<NeuralLayerConfig> layerConfigs)
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
    }
}