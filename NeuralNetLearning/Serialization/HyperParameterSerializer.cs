using System;
using System.Collections.Generic;
using NeuralNetLearning.Maths;
using System.Reflection;
using System.IO;
using System.Text;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetLearning.Serialization
{
    internal class HyperParameterSerializer 
    {
        private static readonly string _simpleHyperParamsFile = "hyper-params.txt";
        private static readonly string _separator = ": ";

        public HyperParameterSerializer()
        { }

        public void WriteToDirectory(object obj, string directory)
        {
            if (!Directory.Exists(directory))
                Directory.CreateDirectory(directory);

            var serializables = from fieldInfo in obj.GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                                let attribute = fieldInfo.GetCustomAttributes(typeof(SerializableHyperParameter), inherit: true)
                                where attribute.Length == 1
                                select (field: fieldInfo.GetValue(obj),
                                        name: (attribute.Single() as SerializableHyperParameter).Name,
                                        type: fieldInfo.FieldType);

            // saving double, int, bool hyper-params to single file
            var simpleSerializables = from s in serializables
                                      let type = s.type
                                      where type.IsEquivalentTo(typeof(double)) ||
                                            type.IsEquivalentTo(typeof(int)) ||
                                            type.IsEquivalentTo(typeof(bool))
                                      select s;
            WriteSimpleSerializablesToFile(obj.GetType(), simpleSerializables, filepath: $"{directory}/{_simpleHyperParamsFile}");


            // saving each Parameter hyper-param to individual file
            var paramSerializables = serializables.Where(s => s.type.IsEquivalentTo(typeof(Parameter)));
            WriteParamSerializablesToDirectory(paramSerializables, directory);
        }

        public TParent ReadFromDirectory<TParent>(string directory)
            where TParent : class
        {
            // reading simple hyper parameters from file
            if (!File.Exists($"{directory}/{_simpleHyperParamsFile}"))
                throw new FileNotFoundException($"Could not find the file ${directory}/{_simpleHyperParamsFile}");

            string[] lines = File.ReadAllLines($"{directory}/{_simpleHyperParamsFile}");

            string typeName = lines.First();
            Type derivedType = Util.GetDerivedTypeWithName(typeof(TParent), typeName);
            Dictionary<string, string> serializedNameToFieldName = SerializedNameToFieldName(derivedType);
            object derivedObject = Activator.CreateInstance(derivedType);

            // setting simple hyper parameter values
            foreach (string line in lines)
            {
                if (line.Contains(_separator))
                {
                    string serializedName = line.Split(_separator)[0];
                    string fieldName = serializedNameToFieldName[serializedName];
                    string serialiedValue = line.Split(_separator)[1];
                    object value = Util.ReadIntBoolDouble(serialiedValue);

                    derivedType
                        .GetField(fieldName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                        .SetValue(derivedObject, value);
                }
            }

            // setting parameter values
            foreach (string paramDir in Directory.EnumerateDirectories(directory))
            {
                string serializedName = paramDir.Split("/").Last().Split("\\").Last();
                string fieldName = serializedNameToFieldName[serializedName];
                Parameter paramValue = ParameterFactory.ReadFromDirectory(paramDir);
                derivedType
                    .GetField(fieldName, BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                    .SetValue(derivedObject, paramValue);
            }

            return derivedObject as TParent;
        }

        private void WriteSimpleSerializablesToFile(Type objType, IEnumerable<(object field, string name, Type type)> simpleSerializables, string filepath)
        {
            StringBuilder stringBuilder = new();
            stringBuilder.AppendLine(objType.Name);
            foreach (var s in simpleSerializables)
            {
                string value = s.field.ToString();
                if (s.type.IsEquivalentTo(typeof(double)) && int.TryParse(value, out int _))
                    value = $"{value}.0";

                stringBuilder.AppendLine($"{s.name}{_separator}{value}");
            }
            File.WriteAllText(filepath, stringBuilder.ToString());
        }

        private static void WriteParamSerializablesToDirectory(IEnumerable<(object field, string name, Type type)> paramSerializables, string directoryPath)
        {
            foreach (var s in paramSerializables)
            {
                string paramDirectory = $"{directoryPath}/{s.name}";
                if (!Directory.Exists(paramDirectory))
                    Directory.CreateDirectory(paramDirectory);

                (s.field as Parameter)?.WriteToDirectory(paramDirectory);
            }
        }

        private static Dictionary<string, string> SerializedNameToFieldName(Type type)
        {
            var serializedNameData = from fieldInfo in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                                     let attribute = fieldInfo.GetCustomAttributes(typeof(SerializableHyperParameter), inherit: true)
                                     where attribute.Length == 1
                                     select (serializedName: (attribute.Single() as SerializableHyperParameter).Name,
                                             fieldName: fieldInfo.Name);

            return serializedNameData.ToDictionary(s => s.serializedName, s => s.fieldName);
        }
    }
}
