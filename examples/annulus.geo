//+
SetFactory("OpenCASCADE");
Circle(1) = {0, 0, 0, 1.0, 0, 2*Pi};
//+
Circle(2) = {0, 0, 0, 2.0, 0, 2*Pi};
//+
Line Loop(1) = {2};
//+
Line Loop(2) = {1};
//+
Plane Surface(1) = {1, 2};
//+
Physical Line("outer_bdy") = {2};
//+
Physical Line("inner_bdy") = {1};
//+
Physical Surface("surface") = {1};
Mesh.CharacteristicLengthMax=0.25 / 1.45;
//Mesh.CharacteristicLengthMin=0.25;

// 449 vertices -- 0.25 / 1.45
// 1718 vertices -- 0.24 / 2.86
