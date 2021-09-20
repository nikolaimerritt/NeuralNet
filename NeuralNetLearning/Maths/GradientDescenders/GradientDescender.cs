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
        /// <summary>
        /// The filepath to which the name of the sub-class of <see cref="GradientDescender"/> is written.
        /// </summary>
        // private static readonly string _nameFile = "gradient-descender-name.txt";
        
        /// <summary>
        /// The filename to which the hyper-parameters of type <see cref="double"/> are written.
        /// </summary>
        protected static readonly string _simpleHyperParamsFile = "hyper-params.txt";

        protected static readonly string _separator = ": ";

        /// <summary>
        /// Returns the value to add to the current <see cref="Parameter"/> in the gradient descent step.
        /// </summary>
        /// <param name="gradient"> The cost gradient </param>
        internal abstract Parameter GradientDescentStep(Parameter gradient);

        /// <summary>
        /// Writes the name of the gradient descender and its hyper-parameter values to <paramref name="directoryPath"/> in a human-readable format.
        /// </summary>
        /// <param name="directoryPath">The (relative or absolute) path to the directory to be written to.</param>
        public abstract void WriteToDirectory(string directoryPath);

        public void ExpWriteToDirectory(string directoryPath)
        {
            if (!Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);

            var serializables = from fieldInfo in this.GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                                let attribute = fieldInfo.GetCustomAttributes(typeof(SerializableHyperParameter), inherit: true)
                                where attribute.Length == 1
                                select (field: fieldInfo.GetValue(this), 
                                        name: (attribute.Single() as SerializableHyperParameter).Name, 
                                        type: fieldInfo.FieldType);

            // saving double, int, bool hyper-params to single file
            var simpleSerializables = from s in serializables
                                      let type = s.type
                                      where type.IsEquivalentTo(typeof(double)) ||
                                            type.IsEquivalentTo(typeof(int)) ||
                                            type.IsEquivalentTo(typeof(bool))
                                      select s;
            WriteSimpleSerializablesToFile(simpleSerializables, filepath: $"{directoryPath}/{_simpleHyperParamsFile}");


            // saving each Parameter hyper-param to individual file
            var paramSerializables = serializables.Where(s => s.type.IsEquivalentTo(typeof(Parameter)));
            WriteParamSerializablesToDirectory(paramSerializables, directoryPath);
        }

        private void WriteSimpleSerializablesToFile(IEnumerable<(object field, string name, Type type)> simpleSerializables, string filepath)
        {
            StringBuilder stringBuilder = new();
            stringBuilder.AppendLine(this.GetType().Name);
            foreach (var s in simpleSerializables)
            {
                string value = s.field.ToString();
                if (s.type.IsEquivalentTo(typeof(double)) && !value.Contains("."))
                    value = $"{value}.0";

                stringBuilder.AppendLine($"{s.name}{_separator}{value}");
            }
            File.WriteAllText(filepath, stringBuilder.ToString());
        }

        private void WriteParamSerializablesToDirectory(IEnumerable<(object field, string name, Type type)> paramSerializables, string directoryPath)
        {
            foreach (var s in paramSerializables)
            {
                string paramDirectory = $"{directoryPath}/{s.name}";
                if (!Directory.Exists(paramDirectory))
                    Directory.CreateDirectory(paramDirectory);

                (s.field as Parameter)?.WriteToDirectory(paramDirectory);
            }
        }

        /// <summary>
        /// Reads the appropriate sub-class of <see cref="GradientDescender"/> that was written to <paramref name="directoryPath"/>.
        /// <para>
        /// The returned sub-class of <see cref="GradientDescender"/> has the same type and hyper-parameter vales as the sub-class of <see cref="GradientDescender"/> that was written.
        /// </para>
        /// </summary>
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
        }

        public static GradientDescender ExReadFromDirectory(string directoryPath)
        {
            Dictionary<string, object> hyperParameters = new();

            // reading simple hyper parameters from file
            if (!File.Exists($"{directoryPath}/{_simpleHyperParamsFile}"))
                throw new FileNotFoundException($"Could not find the file ${directoryPath}/{_simpleHyperParamsFile}");

            string[] lines = File.ReadAllLines($"{directoryPath}/{_simpleHyperParamsFile}");
            
            string typeName = lines.First();
            Type gradType = Util.GetDerivedTypeWithName(typeof(GradientDescender), typeName);
            
            foreach (string line in lines)
            {
                if (line.Contains(_separator))
                {
                    string name = line.Split(_separator)[0];
                    string serialiedValue = line.Split(_separator)[1];
                    object value = Util.ReadIntBoolDouble(serialiedValue);
                    
                    hyperParameters.Add(name, value);
                }
            }

            // reading Parameters from individual files
            foreach (string paramDir in Directory.EnumerateDirectories(directoryPath))
            {
                string name = paramDir.Split("/").Last().Split("\\").Last();
                Parameter param = ParameterFactory.ReadFromDirectory(paramDir);
                hyperParameters.Add(name, param);
            }

            Type type = Util.GetDerivedTypeWithName(typeof(GradientDescender), typeName);
            return Util.CreateInstanceOfType(type, hyperParameters) as GradientDescender;
        }
    }
}
