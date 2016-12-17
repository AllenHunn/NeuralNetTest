using System;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Tectil.NCommand.Contract;

namespace XORNeuralNetTest
{
    public abstract class CommandBase
    {
        public abstract double[][] Input { get; }
        public abstract double[][] Ideal { get; }
        public abstract IMLDataSet TrainingSet { get; }
        public abstract IMLTrain Trainer { get; }
        public abstract BasicNetwork Network { get; }
        public virtual double AcceptableErrorRate { get; } = 0.01;
        public abstract void Execute();

        protected T[][] GenerateArguments<T>(params T[][] args)
        {
            return args;
        }
    }

    public sealed class XorCommand : CommandBase
    {
        public XorCommand()
        {
            Input = GenerateArguments(new[] {0.0, 0.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}, new[] {0.0, 1.0});
            Ideal = GenerateArguments(new[] {0.0}, new[] {1.0}, new[] {0.0}, new[] {1.0});

            Network = new BasicNetwork();
            Network.AddLayer(new BasicLayer(null, true, 2));
            Network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            Network.AddLayer(new BasicLayer(new ActivationElliott(), false, 1));
            Network.Structure.FinalizeStructure();
            Network.Reset();

            TrainingSet = new BasicMLDataSet(Input, Ideal);
            Trainer = new ResilientPropagation(Network, TrainingSet);

            var epoch = 0;
            do
            {
                epoch++;
                Trainer.Iteration();
                Console.WriteLine($"Epoch #: {epoch} Error: {Trainer.Error}");
            } while (Trainer.Error > AcceptableErrorRate);
        }

        public override double[][] Input { get; }

        public override double[][] Ideal { get; }

        public override IMLDataSet TrainingSet { get; }

        public override IMLTrain Trainer { get; }

        public override BasicNetwork Network { get; }

        [Command("Xor")]
        public override void Execute()
        {
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

                var output = Network.Compute(new BasicMLData(new[] {double1, double2}));
                Console.WriteLine($"Values: {double1} {double2} Expected: {expectedOutput} Actual: {output[0]}");
            }
        }
    }

    public sealed class BooleanCommand : CommandBase
    {
        public BooleanCommand()
        {
            Input = GenerateArguments(new[] {0.0, 0.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}, new[] {0.0, 1.0});
            Ideal = GenerateArguments(new[] {0.0}, new[] {1.0}, new[] {1.0}, new[] {1.0});

            Network = new BasicNetwork();
            Network.AddLayer(new BasicLayer(null, true, 2));
            Network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            Network.AddLayer(new BasicLayer(new ActivationElliott(), false, 1));
            Network.Structure.FinalizeStructure();
            Network.Reset();

            TrainingSet = new BasicMLDataSet(Input, Ideal);
            Trainer = new ResilientPropagation(Network, TrainingSet);

            var epoch = 0;
            do
            {
                epoch++;
                Trainer.Iteration();
                Console.WriteLine($"Epoch #: {epoch} Error: {Trainer.Error}");
            } while (Trainer.Error > AcceptableErrorRate);
        }

        public override double[][] Input { get; }

        public override double[][] Ideal { get; }

        public override IMLDataSet TrainingSet { get; }

        public override IMLTrain Trainer { get; }

        public override BasicNetwork Network { get; }

        [Command("Bool")]
        public override void Execute()
        {
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

                var output = Network.Compute(new BasicMLData(new[] {double1, double2}));
                Console.WriteLine($"Values: {double1} {double2} Expected: {expectedOutput} Actual: {output[0]}");
            }
        }
    }

    public sealed class CompareCommand : CommandBase
    {
        public CompareCommand()
        {
            Input = GenerateArguments(new[] {0.0, 0.0}, new[] {1.0, 0.0}, new[] {1.0, 1.0}, new[] {0.0, 1.0},
                new[] {0.5, 0.4}, new[] {0.6, 0.9}, new[] {0.1, 1.0}, new[] {0.5, 1.0}, new[] {0.7, 0.7});

            Ideal = GenerateArguments(new[] {0.0}, new[] {1.0}, new[] {0.0}, new[] {-1.0}, new[] {1.0}, new[] {-1.0},
                new[] {-1.0}, new[] {-1.0}, new[] {0.0});

            Network = new BasicNetwork();
            Network.AddLayer(new BasicLayer(null, true, 2));
            Network.AddLayer(new BasicLayer(new ActivationTANH(), true, 3));
            Network.AddLayer(new BasicLayer(new ActivationTANH(), false, 1));
            Network.Structure.FinalizeStructure();
            Network.Reset();

            TrainingSet = new BasicMLDataSet(Input, Ideal);
            Trainer = new ResilientPropagation(Network, TrainingSet);

            var epoch = 0;
            do
            {
                epoch++;
                Trainer.Iteration();
                Console.WriteLine($"Epoch #: {epoch} Error: {Trainer.Error}");
            } while (Trainer.Error > AcceptableErrorRate);
        }

        public override double[][] Input { get; }

        public override double[][] Ideal { get; }

        public override IMLDataSet TrainingSet { get; }

        public override IMLTrain Trainer { get; }

        public override BasicNetwork Network { get; }

        [Command("Compare")]
        public override void Execute()
        {
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

                var output = Network.Compute(new BasicMLData(new[] {double1, double2}));
                Console.WriteLine($"Values: {double1} {double2} Expected: {expectedOutput} Actual: {output[0]}");
            }
        }
    }
}