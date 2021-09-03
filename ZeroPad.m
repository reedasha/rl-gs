function [Signal] = ZeroPad(Signal, P)
%[ Signal ] = ZeroPad( Signal, P ) 
% Implements delta function upsampling.
% Works for all signal types, real, complex, single or dual pol
%
% Inputs:
% Signal        - signal structure
% P.Ns          - (integer) oversampling
%
% Returns:
% Signal structure with upsampled Et field
%
%
% See also SAMPLEANDHOLD

delta = zeros(1,P.Ns);
delta(1) = 1;
Signal.Et = kron(Signal.Et, delta);

% Adjust the Signal sampling frequency of the signal
Signal.Fs = Signal.Fs*P.Ns;

end
