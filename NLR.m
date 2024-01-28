function [W1,W2,N]=NLR(train_data,train_p_target,optmparameter)

lambda=optmparameter.lambda;
beta=optmparameter.beta;
gamma=optmparameter.gamma;

[n,q]=size(train_p_target);
[~,d]=size(train_data);
max_iter=100;
lf=4;

%initialization
X=train_data;
Y1=train_p_target;
Y2=1-Y1;
W1=zeros(d,q);
W2=zeros(d,q);
N=zeros(n,q);
L=ones(n,q);
M=X'*X;
for iter=1:max_iter

    % W1 subproblem
    W1=(M+beta*M+gamma*eye(d))\(X'*Y1-X'*N+beta*X'*L-beta*M*W2);
 
    % W2 subproblem
    W2=(M+beta*M+gamma*eye(d))\(X'*Y2+X'*N+beta*X'*L-beta*M*W1);

    % N subproblem
    G=N-(1/lf)*(4*N-2*Y1+2*Y2+2*X*W1-2*X*W2);
    N=max(G-lambda/lf,0)+min(G+lambda/lf,0);
    N(N<=0)=0;
    N(N>0)=1;
    N(Y1==0)=0;

end

end

