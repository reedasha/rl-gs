function X=get_MQAM_constellation(M)

pam=[-(sqrt(M)-1):2:sqrt(M)-1];
[A,B]=meshgrid(pam,pam(end:-1:1));
X=A+1i*B;
X=X(:);
Es=1/M*sum(abs(X).^2);
X=X./sqrt(Es);

% % Shuffle the constellation labeling at random
% X = X(randperm(length(X)));

% % Add a bit of noise to the constellations
% X = X+(randn(size(X))+1i*randn(size(X)))/15;
