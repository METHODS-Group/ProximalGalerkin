// To run this file, move it into the examples folder in mfem
// docker run -it --rm -v ./examples/eikonal:/home/euler/shared -w /home/euler/mfem --rm --entrypoint=/bin/bash ghcr.io/methods-group/proximalgalerkin-mfem:main
// cp /home/euler/shared/convert_mesh.cpp /home/euler/mfem/examples/
// make convert_mesh
// ./convert_mesh --mesh ../data/mobius-strip.mesh
// cp -r  mobius-strip.mesh/ ../../shared/

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;
int main(int argc, char *argv[])
{

   const char *mesh_file = ".mesh";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to visualize.");
   args.Parse();
   if (!args.Good())
   {
      if (!args.Help())
      {
         args.PrintError(cout);
         cout << endl;
      }
      cout << "Visualize and manipulate a serial mesh:\n"
           << "   mesh-explorer -m <mesh_file>\n"
           << "Visualize and manipulate a parallel mesh:\n"
           << "   mesh-explorer -np <#proc> -m <mesh_prefix>\n"
           << endl
           << "All Options:\n";
      args.PrintHelp(cout);
      return 1;
   }
   args.PrintOptions(cout);
   Mesh *mesh;
   mesh = new Mesh(mesh_file, 1, 0);
   mesh->SetCurvature(3);
   ParaViewDataCollection *pd = NULL;
   pd = new ParaViewDataCollection(mesh_file, mesh);
   pd->SetPrefixPath("");
   pd->SetLevelsOfDetail(3);
   pd->SetDataFormat(VTKFormat::ASCII);
   pd->SetHighOrderOutput(true);
   pd->SetCycle(0);
   pd->SetTime(0.0);
   pd->Save();
}
