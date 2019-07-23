SetFactory("OpenCASCADE");
Merge "circle_in_square.brep";

// This line (and the SetFactory) merge duplicate lines created by CAD,
// very imporant!
Coherence;

//+
Physical Line("scatterer") = {11};
//+
Physical Line("outer_boundary") = {18, 17, 16, 2, 23, 22, 20, 24, 7, 25, 12, 14};
//+
Physical Surface("inner_region") = {3};
//+
Physical Surface("pml_x_region") = {2, 1};
//+
Physical Surface("pml_y_region") = {6, 7};
//+
Physical Surface("pml_xy_region") = {5, 8, 9, 4};
