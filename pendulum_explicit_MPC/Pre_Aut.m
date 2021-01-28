function PreS=Pre_Aut(Acl,S)
% Compute pre set for autonomous system. Acl is the closed-loop A matrix,
% and S is the target matrix. S is the constraint polytope on states and 
% input.
% works with polytope which are also not full dimensional
nx=size(Acl,2);
PreS=Polyhedron('H',[S.H(:,1:nx)*Acl S.H(:,nx+1)],...
'He',[S.He(:,1:nx)*Acl S.He(:,nx+1)]);
end