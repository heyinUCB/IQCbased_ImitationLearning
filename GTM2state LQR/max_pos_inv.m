function [Oinf,converged]=max_pos_inv(Acl,S)
% Acl is the closed-loop A matrix, it is opten written as Acl = A+B*F. S is
% the constraint polytope on states and input.
maxIterations=500;
Omega_i = S; % initialization
for i = 1:maxIterations
    % compute backward reachable set
    P = Pre_Aut(Acl,Omega_i);
    % intersect with the state constraints
    P = P.intersect(Omega_i).minHRep();
    if P==Omega_i
        Oinf=Omega_i;
        break
    else
        Omega_i = P;
    end
end
if i==maxIterations,
    converged=0;
else
    converged=1;
end
end