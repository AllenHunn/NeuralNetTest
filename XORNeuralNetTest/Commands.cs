using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Specific;
using Encog.ML.Train;
using Encog.Neural;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Tectil.NCommand.Contract;

namespace XORNeuralNetTest
{
    public class Commands
    {
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
            int epoch = 0;
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

        private static double[][] XorInput => new[] { new[] { 0.0, 0.0 }, new[] { 1.0, 0.0 }, new[] { 0.0, 1.0 }, new[] { 1.0, 1.0 } };

        private static double[][] XorIdeal => new[] { new[] { 0.0 }, new[] { 1.0 }, new[] { 1.0 }, new[] { 0.0 } };

        [Command]
        public void Add()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            network.AddLayer(new BasicLayer(new ActivationElliott(), false, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            IMLDataSet trainingSet = new BasicMLDataSet(AddInput, AddIdeal);
            IMLTrain train = new ResilientPropagation(network, trainingSet);
            int epoch = 0;
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

                var output = network.Compute(new BasicMLData(new[] { double1, double2 }));
                Console.WriteLine($"Values: {double1} {double2} Expected: {expectedOutput} Actual: {output[0]}");
            }
        }

        private static double[][] AddInput => new[] { new[] { 1.0, 1.0 }, new[] { 5.0, 10.0 }, new[] { 4.0, 2.0 }, new[] { 2.0, 2.0 }, new[] { 3.0, 5.0 }, new[] { 1.0, 2.0 }, new[] { 5.0, 11.0 }, new[] { 4.0, 3.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 6.0 } };

        private static double[][] AddIdeal => new[] { new[] { 2.0 }, new[] { 15.0 }, new[] { 6.0 }, new[] { 4.0 }, new[] { 8.0 }, new[] { 3.0 }, new[] { 16.0 }, new[] { 7.0 }, new[] { 5.0 }, new[] { 9.0 } };
    }
}