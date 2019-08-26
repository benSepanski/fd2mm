Merge "ball_in_cube.brep";
Point(11) = {-2, -2, -2, 1.0};
Point(12) = {-2, -2, 2, 1.0};
Point(13) = {-2, 2, -2, 1.0};
Point(14) = {-2, 2, 2, 1.0};
Point(15) = {2, -2, -2, 1.0};
Point(16) = {2, -2, 2, 1.0};
Point(17) = {2, 2, -2, 1.0};
Point(18) = {2, 2, 2, 1.0};
//+
Line(16) = {13, 14};
//+
Line(17) = {14, 18};
//+
Line(18) = {18, 16};
//+
Line(19) = {16, 12};
//+
Line(20) = {12, 14};
//+
Line(21) = {12, 11};
//+
Line(22) = {11, 13};
//+
Line(23) = {13, 17};
//+
Line(24) = {17, 15};
//+
Line(25) = {15, 11};
//+
Line(26) = {17, 18};
//+
Line(28) = {16, 15};
//+
Line Loop(8) = {23, 26, -17, -16};
//+
Plane Surface(8) = {8};
//+
Line Loop(9) = {22, 16, -20, 21};
//+
Plane Surface(9) = {9};
//+
Line Loop(10) = {25, -21, -19, 28};
//+
Plane Surface(10) = {10};
//+
Line Loop(11) = {18, 19, 20, 17};
//+
Plane Surface(11) = {11};
//+
Line Loop(12) = {26, 18, 28, -24};
//+
Plane Surface(12) = {12};
//+
Line Loop(13) = {23, 24, 25, 22};
//+
Plane Surface(13) = {13};
//+
Physical Surface("inner_bdy") = {7};
//+
Physical Surface("pml_bdy") = {8, 13, 9, 12, 10, 11};
//+
Physical Surface("outer_bdy") = {4, 3, 1, 5, 6, 2};
//+
Surface Loop(3) = {8, 13, 9, 12, 10, 11};
//+
Surface Loop(4) = {7};
//+
Surface Loop(5) = {4, 3, 1, 5, 6, 2};
//+
Volume(2) = {3, 4};
//+
Volume(3) = {5, 3};
//+
Physical Volume("inner_region") = {2};
//+
Physical Volume("pml_region") = {3};
