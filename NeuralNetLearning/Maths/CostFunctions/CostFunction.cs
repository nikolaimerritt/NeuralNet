using MathNet.Numerics.LinearAlgebra;
using NeuralNetLearning.Serialization;
using System.Collections.Generic;
using NeuralNetLearning;
using System.Linq;
using System;
using System.IO;

namespace NeuralNetLearning.Maths.CostFunctions
{
    /// <summary>
    /// Represents a cost function with vector arguments.
    /// </summary>
    public abstract class CostFunction
    {
        /// <summary>
        /// Returns the numeric cost of the calculated vector <paramref name="predicted"/> vs the target vector <paramref name="expected"/>.
        /// </summary>
        public abstract double Apply(Vector<double> predicted, Vector<double> expected);
        
        /// <summary>
        /// Returns the vector of derivatives of the cost function with respect to the calculated vector <paramref name="predicted"/>.
        /// </summary>
        public abstract Vector<double> Derivative(Vector<double> predicted, Vector<double> expected);

        /// <summary>
        /// Writes the name of the cost function and its hyper-parameters to <paramref name="filepath"/>.
        /// </summary>

        /// <summary>
        /// Reads the appropriate sub-class of <see cref="CostFunction"/> that was written to <paramref name="filepath"/>.
        /// <para>
        /// The returned sub-class of <see cref="CostFunction"/> has the same type and hyper-parameter vales as the sub-class of <see cref="CostFunction"/> that was written.
        /// </para>
        /// </summary>
    }
}
