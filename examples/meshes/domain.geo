a = 2;
b = 3;
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {-1, 0, 0, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Circle(1) = {3, 1, 2};
//+
Circle(2) = {2, 1, 3};

//+
Point(4) = {a, -a, 0, 1.0};
//+
Point(5) = {a, a, 0, 1.0};
//+
Point(6) = {-a, a, 0, 1.0};
//+
Point(7) = {-a, -a, 0, 1.0};
//+
Line(3) = {4, 5};
//+
Line(4) = {5, 6};
//+
Line(5) = {6, 7};
//+
Line(6) = {7, 4};

//+
Point(8) = {b, -b, 0, 1.0};
//+
Point(9) = {b, -a, 0, 1.0};
//+
Point(10) = {b, a, 0, 1.0};
//+
Point(11) = {b, b, 0, 1.0};
//+
Point(12) = {a, b, 0, 1.0};
//+
Point(13) = {-a, b, 0, 1.0};
//+
Point(14) = {-b, b, 0, 1.0};
//+
Point(15) = {-b, a, 0, 1.0};
//+
Point(16) = {-b, -a, 0, 1.0};
//+
Point(17) = {-b, -b, 0, 1.0};
//+
Point(18) = {-a, -b, 0, 1.0};
//+
Point(19) = {a, -b, 0, 1.0};

//+
Line(7) = {8, 9};
//+
Line(8) = {9, 10};
//+
Line(9) = {10, 11};
//+
Line(10) = {11, 12};
//+
Line(11) = {12, 13};
//+
Line(12) = {13, 14};
//+
Line(13) = {14, 15};
//+
Line(14) = {15, 16};
//+
Line(15) = {16, 17};
//+
Line(16) = {17, 18};
//+
Line(17) = {18, 19};
//+
Line(18) = {19, 8};
//+
Line(19) = {4, 19};
//+
Line(20) = {4, 9};
//+
Line(21) = {5, 12};
//+
Line(22) = {5, 10};
//+
Line(23) = {6, 13};
//+
Line(24) = {6, 15};
//+
Line(25) = {7, 16};
//+
Line(26) = {7, 18};
//+
Line Loop(27) = {3, 4, 5, 6};
//+
Line Loop(28) = {1, 2};
//+
Plane Surface(29) = {27, 28};
//+
Line Loop(30) = {6, 19, -17, -26};
//+
Plane Surface(31) = {30};
//+
Line Loop(32) = {8, -22, -3, 20};
//+
Plane Surface(33) = {32};
//+
Line Loop(34) = {4, 23, -11, -21};
//+
Plane Surface(35) = {34};
//+
Line Loop(36) = {5, 25, -14, -24};
//+
Plane Surface(37) = {36};
//+
Line Loop(38) = {20, -7, -18, -19};
//+
Plane Surface(39) = {38};
//+
Line Loop(40) = {22, 9, 10, -21};
//+
Plane Surface(41) = {40};
//+
Line Loop(42) = {23, 12, 13, -24};
//+
Plane Surface(43) = {42};
//+
Line Loop(44) = {25, 15, 16, -26};
//+
Plane Surface(45) = {44};
//+
Physical Surface("pml_x_region") = {33, 37};
//+
Physical Surface("pml_y_region") = {35, 31};
//+
Physical Surface("pml_xy_region") = {41, 39, 45, 43};
//+
Physical Surface("inner_region") = {29};
//+
Physical Line("inner_bdy") = {1, 2};
//+
Physical Line("outer_bdy") = {11, 12, 13, 14, 15, 16, 17, 18, 7, 8, 9, 10};
//+
Mesh.CharacteristicLengthMax=0.125;
//Mesh.CharacteristicLengthMin=0.15;
