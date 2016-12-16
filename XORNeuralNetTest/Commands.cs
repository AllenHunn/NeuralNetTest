using System;
using System.Linq;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Util.Arrayutil;
using Tectil.NCommand.Contract;

namespace XORNeuralNetTest
{
    public class Commands
    {
        private static double[][] XorInput
            => new[] {new[] {0.0, 0.0}, new[] {1.0, 0.0}, new[] {0.0, 1.0}, new[] {1.0, 1.0}};

        private static double[][] XorIdeal => new[] {new[] {0.0}, new[] {1.0}, new[] {1.0}, new[] {0.0}};

        private static double[][] AddInput
            =>
                new[]
                {
                    new[] {1.0, 1.0}, new[] {5.0, 10.0}, new[] {4.0, 2.0}, new[] {2.0, 2.0}, new[] {3.0, 5.0},
                    new[] {1.0, 2.0}, new[] {5.0, 11.0}, new[] {4.0, 3.0}, new[] {2.0, 3.0}, new[] {3.0, 6.0},
                    new[] {2.0, 2.0}, new[] {6.0, 11.0}, new[] {5.0, 3.0}, new[] {3.0, 3.0}, new[] {4.0, 6.0}
                };

        private static double[][] AddIdeal
            =>
                new[]
                {
                    new[] {2.0}, new[] {15.0}, new[] {6.0}, new[] {4.0}, new[] {8.0}, new[] {3.0}, new[] {16.0},
                    new[] {7.0}, new[] {5.0}, new[] {9.0}, new[] {4.0}, new[] {17.0},
                    new[] {8.0}, new[] {6.0}, new[] {10.0}
                };

        [Command]
        public void Xor()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            network.AddLayer(new BasicLayer(new ActivationElliott(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            IMLDataSet trainingSet = new BasicMLDataSet(XorInput, XorIdeal);
            IMLTrain train = new ResilientPropagation(network, trainingSet);
            var epoch = 0;
            do
            {
                epoch++;
                train.Iteration();
                Console.WriteLine($"Epoch #: {epoch} Error: {train.Error}");
            } while (train.Error > 0.01);

            while (true)
            {
                Console.WriteLine("Please enter two double values (quit or q to quit):");
                var input1 = Console.ReadLine()?.ToLower() ?? "quit";
                if (input1 == "quit" || input1 == "q")
                {
                    break;
                }

                var double1 = double.Parse(input1);

                var input2 = Console.ReadLine()?.ToLower() ?? "quit";
                if (input2 == "quit" || input2 == "q")
                {
                    break;
                }

                var double2 = double.Parse(input2);

                Console.WriteLine("Now enter expected result as double");
                var inputOut = Console.ReadLine();
                var expectedOutput = double.Parse(inputOut);

                var output = network.Compute(new BasicMLData(new[] {double1, double2}));
                Console.WriteLine($"Values: {double1} {double2} Expected: {expectedOutput} Actual: {output[0]}");
            }
        }

        [Command]
        public void Add()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            network.AddLayer(new BasicLayer(new ActivationElliott(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            var normalizer = new NormalizeArray(0, 1);

            var inputMax = AddInput.SelectMany(x => x).Max();
            var idealMax = AddIdeal.SelectMany(x => x).Max();

            var normalizedInput = (double[][]) AddInput.Clone();
            var normalizedIdeal = (double[][]) AddIdeal.Clone();
            var allNormalizedInput = normalizer.Process(normalizedInput.SelectMany(x => x).ToArray());
            var allNormalizedIdeal = normalizer.Process(normalizedIdeal.SelectMany(x => x).ToArray());

            for (var i = 0; i < normalizedInput.Length; i++)
            {
                normalizedInput[i] = allNormalizedInput.Skip(i*2).Take(2).ToArray();
                normalizedIdeal[i] = allNormalizedIdeal.Skip(i).Take(1).ToArray();
            }

            IMLDataSet trainingSet = new BasicMLDataSet(normalizedInput, normalizedIdeal);
            IMLTrain train = new ResilientPropagation(network, trainingSet);
            var epoch = 0;
            do
            {
                epoch++;
                train.Iteration();
                Console.WriteLine($"Epoch #: {epoch} Error: {train.Error}");
            } while (train.Error > 0.01);

            while (true)
            {
                Console.WriteLine("Please enter two double values (quit or q to quit):");
                var input1 = Console.ReadLine()?.ToLower() ?? "quit";
                if (input1 == "quit" || input1 == "q")
                {
                    break;
                }

                var double1 = double.Parse(input1);

                var input2 = Console.ReadLine()?.ToLower() ?? "quit";
                if (input2 == "quit" || input2 == "q")
                {
                    break;
                }

                var double2 = double.Parse(input2);

                Console.WriteLine("Now enter expected result as double");
                var inputOut = Console.ReadLine();
                var expectedOutput = double.Parse(inputOut);

                var output = network.Compute(new BasicMLData(new[] {double1, double2}));
                Console.WriteLine($"Values: {double1} {double2} Expected: {expectedOutput} Actual: {output[0]}");
            }
        }
    }
}