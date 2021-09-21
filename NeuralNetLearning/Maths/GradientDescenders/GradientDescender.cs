using NeuralNetLearning.Serialization;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Linq;
using System.Text;
using System;
using System.IO;

namespace NeuralNetLearning.Maths.GradientDescenders
{
    public abstract class GradientDescender
    {
        //private static readonly HyperParameterSerializer<GradientDescender> _serializer = new(simpleHyperParamsFile: "hyper-params");
        /// <summary>
        /// The filepath to which the name of the sub-class of <see cref="GradientDescender"/> is written.
        /// </summary>
        // private static readonly string _nameFile = "gradient-descender-name.txt";
        
        /// <summary>
        /// The filename to which the hyper-parameters of type <see cref="double"/> are written.
        /// </summary>

        /// <summary>
        /// Returns the value to add to the current <see cref="Parameter"/> in the gradient descent step.
        /// </summary>
        /// <param name="gradient"> The cost gradient </param>
        internal abstract Parameter GradientDescentStep(Parameter gradient);

        /// <summary>
        /// Writes the name of the gradient descender and its hyper-parameter values to <paramref name="directoryPath"/> in a human-readable format.
        /// </summary>
        /// <param name="directoryPath">The (relative or absolute) path to the directory to be written to.</param>

        /// <summary>
        /// Reads the appropriate sub-class of <see cref="GradientDescender"/> that was written to <paramref name="directoryPath"/>.
        /// <para>
        /// The returned sub-class of <see cref="GradientDescender"/> has the same type and hyper-parameter vales as the sub-class of <see cref="GradientDescender"/> that was written.
        /// </para>
        /// </summary>
        /// 
        /*
        public static GradientDescender ReadFromDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException($"Could not find directory {directoryPath}");

            string typeName = HyperParamEncoder.DecodeTypeName($"{directoryPath}/{_simpleHyperParamsFile}");
            if (typeName == nameof(StochasticGradientDescender))
                return StochasticGradientDescender.Read(directoryPath);

            if (typeName == nameof(AdamGradientDescender))
                return AdamGradientDescender.Read(directoryPath);

            else throw new Exception($"Could not recognise the gradient descender with type name {typeName}");
        } */

        
    }
}
