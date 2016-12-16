using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tectil.NCommand;

namespace XORNeuralNetTest
{
    class Program
    {
        static void Main(string[] args)
        {
            NCommands commands = new NCommands();
            commands.Context.AutodetectCommandAssemblies();
            commands.Context.Configuration.DisplayExceptionDetails = false;
            commands.RunConsole(args);
        }
    }
}
