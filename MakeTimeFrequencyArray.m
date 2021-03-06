function [varargout] = MakeTimeFrequencyArray(Signal)

% Creates a continuous wave signal generated by a CW laser with phase noise
%
% [FF,{TT}] = MakeTimeFrequencyArray(Signal)
% Output TT is optional
%
% Inputs:
% Signal          - Signal structure
% Signal.Fs       - Sampling rate [Hz]
% Signal.Et       - Field
%
% Returns:
% FF         - Frequency array [Hz]
% TT         - Time array [s]
%

[~,Nt] = size(Signal.Et);
dT = 1/Signal.Fs;                                   % Temporal resolution (s)
dF = 1/Nt/dT;                                       % Frequency resolution (Hz)
TT = (0:Nt-1) * dT;                                 % Time array (s)
FF = [0:floor(Nt/2)-1,floor(-Nt/2):-1] * dF;        % Frequency array (Hz)

varargout{1} = FF;
varargout{2} = TT;

end

