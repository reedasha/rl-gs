function Signal=AddNoise(Signal,P)

% Add complex white gaussian noise to a noise free optical signal to a give specified OSNR
% for a noise equivalent bandwidth specified or if not specified a default
% value of 12.5 GHz is used.
%
% The noise is added equally to both polarisation states
%
% Signal=AddNoise(Signal,P)
%
% Required inputs:
% SignalIn - input signal structure
% P.OSNR - (dB) required optical signal-to-noise ratio
%
% Optional inputs:
% P.NoiseBW - (Hz) noise bandwidth for calculation of OSNR; if not
%                 specified P.NoiseBW is 12.5 GHz
% P.SigPower - (dBm) signal Power
%
% Returns:
% Signal - output signal structure
%
% Author: Benn Thomsen, May 2005.
% Modified by: Heribert Brust, December 2005
% Mod. D Millar May 2008:
% Mod. S Kilmurray Nov 2013:

[Np,Nt]=size(Signal.Et);

if isfield(Signal,'Fchan')
    N_WDM=length(Signal.Fchan);
else
    N_WDM=1;                         % (number of WDM channels) default value
end

if ~isfield(P,'NoiseBW'),
    P.NoiseBW=12.5e9;              % (Hz) default value    
end

if ~isfield(P,'SigPower'),                          %% if signal Power must be calculated
    sigPower = sum(sum((real(Signal.Et).^2+imag(Signal.Et).^2),1),2)/Nt/N_WDM; % Average Signal Power (W) over sim BW
else
    sigPower = 10^((P.SigPower-30)/10);             %% signal Power from input
end

noisePower = sigPower*10^(-P.OSNR/10);              % Noise Power over simulation bandwidth (W)
noisePower = noisePower*Signal.Fs/P.NoiseBW;        % Noise Power over Ref reference bandwidth (W)

% check number of polarisation states if only 1 then create other state
if Np == 1,
    Signal.Et = [Signal.Et; zeros(1, Nt)];
end

% Additive complex white Gaussian noise (2 polarisation states) zero mean and standard deviation corresponding
% to the noise Power evenly split between dimensions
gaussianNoise = sqrt(0.25*noisePower)*(randn(2,Nt)+1i*randn(2,Nt));

Signal.Et = Signal.Et + gaussianNoise;

