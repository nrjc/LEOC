%% inverted_pendulum.m

% Copyright (C) 2020 by
% Zhang Naifu and Nicholas Capel.
%
% Last modified: 2020-03-10
%

% function [A, B, C, D] = inverted_pendulum(M, m, b, I, l)
%% Code

M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
l = 0.3;
g = 9.8;

p = I*(M+m)+M*m*l^2; % denominator for the A and B matrices

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];
C = [1 0 0 0;
     0 0 1 0];
D = [0;
     0];

states = {'x' 'x_dot' 'phi' 'phi_dot'};
inputs = {'u'};
outputs = {'x'; 'phi'};

sys_ss = ss(A,B,C,D,'statename',states,'inputname',inputs,'outputname',outputs)
A = sys_ss.A; B = sys_ss.B; C = sys_ss.C; D = sys_ss.D;