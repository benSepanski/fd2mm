Merge "betterplane_pml.brep";

//+
Surface Loop(4) = {10, 11, 7, 9, 8, 12};  // PML Bdy Loop
//+ 
Surface Loop(5) = {23, 29, 35, 18, 13, 14, 20, 25, 31, 15, 16, 21, 26, 32, 37, 40, 27, 33, 38, 41, 28, 34, 39, 42, 17, 22, 19, 24, 30, 36};  // Plane surface loop
//+
Surface Loop(6) = {4, 5, 1, 3, 2, 6}; // Outer Bdy Loop
//+
Volume(4) = {6, 5}; // Inner region
//+
Volume(5) = {4, 6};  // PML region
//+
Physical Surface("outer_bdy") = {4, 1, 3, 6, 5, 2};
//+
Physical Surface("scatterer_bdy") = {21, 13, 42, 34, 33, 38, 27, 26, 32, 40, 28, 20, 23, 17, 15, 29, 35, 31, 36, 24, 39, 19, 18, 14, 22, 30, 25, 37, 41};
//+
Physical Volume("inner_region") = {4};
//+
Physical Volume("pml_region") = {5};
