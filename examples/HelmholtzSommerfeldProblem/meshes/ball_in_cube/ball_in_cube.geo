SetFactory("OpenCASCADE");
Merge "ball_in_cube.brep";
//+
Physical Surface("outer_boundary") = {4, 1, 3, 2, 6, 5};
//+
Physical Surface("scatterer_boundary") = {7};
//+
Physical Volume("inner_region") = {1};
