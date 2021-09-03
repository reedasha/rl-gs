function [ Symbols ] = SymbolAlignment( Et, P )
% Calculates the timing and phase skew (+/- n*pi/2) between the input
% symbols and the transmitted symbol sequence.
%
% Symbols = SymbolAlignment(Et, P)
%
% Inputs:
% Et - Two polarisation vector
% P.ModSymbols - Transmitted symbol vector
%
% Domanic Lavery, June 2014

%% Symbol-to-Integer Mapping
Symbols = zeros(size(Et))+1i*zeros(size(Et));
%% Test to maximise correlation on both polarisation (overcomes polarisation ambiguity)
[xcorrT1]=xcorr(Et(1,:)-mean(Et(1,:)),P.TransmittedSymbols(1,:));
[xcorrT2]=xcorr(Et(1,:)-mean(Et(1,:)),P.TransmittedSymbols(2,:));

[~,loc1]=max([max(abs(xcorrT1)) max(abs(xcorrT2))]);

if(loc1-1) %#ok % (loc1-1) = [0,1], therefore logical
    Et = fliplr(Et.').';
    if(isfield(P,'verbose')&&(P.verbose>1)),disp('Flip XY in BER counter'), end
end

%% Decode on this constellation
for index = 1:size(Et,1)
    % Cross correlate received vector with transmitted vector and find symbol-wise delay
    [xcorr1,lag1]=xcorr(Et(index,:)-mean(Et(index,:)),P.TransmittedSymbols(index,:));
    [~,loc1]=max(xcorr1);
    
    % Correction for phase rotation; round rotation to nearest pi/2
    rotangle = pi/2*round(angle(xcorr1(loc1))/(pi/2));
    
    % Derotate 'received' vector to synchronise with transmitted symbols
    Symbols(index,:) = exp(-1i*(rotangle))*circshift(Et(index,:),[0 -lag1(loc1)]);
end

end

