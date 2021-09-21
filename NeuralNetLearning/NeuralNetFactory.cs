using MathNet.Numerics.LinearAlgebra;
using NeuralNetLearning.LayerConfig;
using NeuralNetLearning.Maths;
using NeuralNetLearning.Maths.Activations;
using NeuralNetLearning.Maths.CostFunctions;
using NeuralNetLearning.Maths.GradientDescenders;
using System;
using System.Collections.Generic;
using NeuralNetLearning.Serialization;
using System.IO;
using System.Linq;


namespace NeuralNetLearning
{
    using static ParameterFactory;
    using Vector = Vector<double>;

    public static class NeuralNetFactory
    {
        /// <summary>
        /// The name of the file in which the <see cref="CostFunction"/> used by a <see cref="NeuralNet"/> is recorded when writing a <see cref="NeuralNet"/> to a directory.
        /// </summary>
        public static readonly string CostFolder = "cost";
        /// <summary>
        /// The name of the folder in which the <see cref="Parameter"/> used by a <see cref="NeuralNet"/> is recorded when writing a <see cref="NeuralNet"/> to a directory.
        /// </summary>
        public static readonly string ParamsFolder = "parameters";
        /// <summary>
        /// The name of the folder in which the <see cref="Activation"/>s used by a <see cref="NeuralNet"/> are recorded when writing a <see cref="NeuralNet"/> to a directory.
        /// </summary>
        public static readonly string ActivationsFolder = "activations";
        /// <summary>
        /// The name of the folder in which the <see cref="GradientDescender"/> used by a <see cref="NeuralNet"/> is recorded when writing a <see cref="NeuralNet"/> to a directory.
        /// </summary>
        public static readonly string GradientDescenderFolder = "gradient-descender";

        private static readonly HyperParameterSerializer _serializer = new();

        /// <summary>
        /// Returns a random <see cref="NeuralNet"/> that is optimised for the use of <see cref="TanhActivation"/>. 
        /// <para>
        /// Uses <see href="https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">Xavier initialisation</see>.
        /// </para>
        /// </summary>
        /// <param name="layerStructure"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet OptimisedForTanh(IList<NeuralLayerConfig> layerStructure, GradientDescender gradientDescender, CostFunction cost)
        {
            Parameter param = XavierInit(LayerSizesFromConfigs(layerStructure));
            Activation[] activators = ActivationsFromConfigs(layerStructure);

            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        /// <summary>
        /// Returns a random <see cref="NeuralNet"/> that is optimised for the use of <see cref="ReluActivation"/>. 
        /// <para>
        /// Uses <see href="https://arxiv.org/abs/1502.01852v1">Kaiming He</see> initialisation.
        /// </para>
        /// </summary>
        /// <param name="layerStructure"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet OptimisedForRelu(IList<NeuralLayerConfig> layerStructure, GradientDescender gradientDescender, CostFunction cost)
        {
            Parameter param = KaimingInit(LayerSizesFromConfigs(layerStructure));
            Activation[] activations = ActivationsFromConfigs(layerStructure);

            return new NeuralNet(param, activations, gradientDescender, cost);
        }

        /// <summary>
        /// Returns a random <see cref="NeuralNet"/> that is optimised for learning <paramref name="trainingData"/>.
        /// <para>
        /// Uses <see href="http://cmp.felk.cvut.cz/~mishkdmy/papers/mishkin-iclr2016.pdf">LSUV initialisation</see>.
        /// </para>
        /// </summary>
        /// <param name="layerStructure"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="trainingData"> The training set which the NeuralNet will be optimised to learn. 
        /// <para> Only the inputs are needed, so this method simply calls its overload where <paramref name="trainingData"/> is a list of inputs.</para>
        /// </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet OptimisedForTrainingData(IList<NeuralLayerConfig> layerStructure, IEnumerable<(Vector input, Vector desiredOutput)> trainingData, GradientDescender gradientDescender, CostFunction cost)
        {
            IList<Vector> miniBatchInputs = trainingData.Select(pair => pair.input).ToList();
            return OptimisedForTrainingData(layerStructure, miniBatchInputs, gradientDescender, cost);
        }

        /// <summary>
        /// Returns a random <see cref="NeuralNet"/> that is optimised for learning <paramref name="trainingData"/>.
        /// <para>
        /// Uses <see href="http://cmp.felk.cvut.cz/~mishkdmy/papers/mishkin-iclr2016.pdf">LSUV initialisation</see>.
        /// </para>
        /// </summary>
        /// <param name="layerStructure"> The configurations of the layer that the NeuralNet will conform to. </param>
        /// <param name="trainingInputs"> The inputs of the training set which the NeuralNet will be optimised to learn. </param>
        /// <param name="gradientDescender"> The gradient descender that the NeuralNet will use when executing gradient descent. </param>
        /// <param name="cost"> The cost function that the NeuralNet will use when executing gradient descent. </param>
        public static NeuralNet OptimisedForTrainingData(IList<NeuralLayerConfig> layerStructure, IEnumerable<Vector> trainingInputs, GradientDescender gradientDescender, CostFunction cost)
        {
            Activation[] activators = ActivationsFromConfigs(layerStructure);
            Parameter param = LSUVInit(LayerSizesFromConfigs(layerStructure), activators, trainingInputs);
            
            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        /// <summary>
        /// Reads the <see cref="NeuralNet"/> object that has been written to <paramref name="directoryPath"/> using the function <see cref="NeuralNet.WriteToDirectory(string)"/>.
        /// <para>
        /// The returned <see cref="NeuralNet"/> has equivalent <see cref="Parameter"/> values, <see cref="Activation"/>, <see cref="GradientDescender"/> and <see cref="CostFunction"/> 
        /// compared to the written <see cref="NeuralNet"/>.
        /// </para>
        /// </summary>
        /// <param name="directoryPath"> The (absolute or relative) path to which a Neural Net has been written. </param>
        /// <returns></returns>
        public static NeuralNet ReadFromDirectory(string directoryPath)
        {
            Parameter param = ParameterFactory.ReadFromDirectory($"{directoryPath}/{ParamsFolder}");
            Activation[] activators = ReadActivationsFromDirectory($"{directoryPath}/{ActivationsFolder}");
            GradientDescender gradientDescender = _serializer.ReadFromDirectory<GradientDescender>($"{directoryPath}/{GradientDescenderFolder}");
            CostFunction cost = _serializer.ReadFromDirectory<CostFunction>($"{directoryPath}/{CostFolder}");

            return new NeuralNet(param, activators, gradientDescender, cost);
        }

        private static Activation[] ReadActivationsFromDirectory(string directory)
        {
            if (!Directory.Exists(directory))
                throw new FileNotFoundException($"Could not find directory {directory}");

            List<string> activationDirectories = Directory.GetDirectories(directory)
                .Where(dir => dir.Contains("activation"))
                .ToList();
            activationDirectories.Sort();

            return activationDirectories
                .Select(_serializer.ReadFromDirectory<Activation>)
                .ToArray();
        }


        private static int[] LayerSizesFromConfigs(IList<NeuralLayerConfig> layerConfigs)
                => layerConfigs
                    .Select(l => l.Size)
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

                activators.Add((layerConfigs[i] as HiddenLayer).Activation);
            }

            if (!(layerConfigs.Last() is OutputLayer))
                throw new ArgumentException($"Expected the last layer to be of type {typeof(OutputLayer)}");

            activators.Add((layerConfigs.Last() as OutputLayer).Activation);

            return activators.ToArray();
        }
    }
}