using System;
using System.IO;
using System.Linq;

namespace NeuralNetLearning.Maths
{
    public class HyperParamEncoder
    {
        public HyperParamEncoder() { }

        public static string Encode(string typeName, params (string, double)[] namesWithVals)
        {
            string encoding = typeName;
            foreach ((string name, double val) in namesWithVals)
            {
                encoding += $"\n{name}: {val}";
            }
            return encoding;
        }

        public static void EncodeToFile(string typeName, string filepath, params (string, double)[] namesWithVals)
            => File.WriteAllText(filepath, Encode(typeName, namesWithVals));

        public static string DecodeTypeName(string filepath)
            => File.ReadAllLines(filepath).First();

        public static double Decode(string filepath, string paramName)
        {
            var lines = File.ReadAllLines(filepath);
            if (!lines.Any())
                throw new Exception($"The file at {filepath} is empty");

            var paramLines = lines.Where(line => line.StartsWith(paramName));
            if (!paramLines.Any())
                throw new Exception($"No parameters with name {paramName} could be found in file {filepath}");

            if (paramLines.Count() > 1)
                throw new Exception($"Multiple parameters have the same name {paramName} in the file {filepath}");

            return Double.Parse(paramLines.Single().Split(":")[1].Trim());
        }   
    }
}
