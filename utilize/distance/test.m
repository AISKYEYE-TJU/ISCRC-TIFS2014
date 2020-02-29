function test

A = rand(10, 20);
B = rand(10, 30);

M = slmetric_pw(A, B, 'eucdist')