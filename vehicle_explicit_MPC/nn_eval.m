function u = nn_eval(W1,W2,W3,x)
W{1} = W1;
W{2} = W2;
W{3} = W3;
z = x;
for i = 1:2
    z = W{i}*z;
    z = tanh(z);
end
u = W{end}*z;
end
