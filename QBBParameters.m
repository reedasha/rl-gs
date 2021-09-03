function [ varargout ] = QBBParameters( M, P)

%% Signal Parameters
P.M         = M;                                    % Number of constellation points
ModFormat   = [num2str(P.M) 'QAM'];                 % Modulation format string
P.ModFormat = ModFormat;                            % Must specify. Square QAM only: ['QPSK','16QAM','64QAM', &c.]
P.Fb        = 8e9;                                  % Channel symbol rate [Hz]
P.Fc        = 3e8/1550e-9;                          % Center frequency [Hz]
P.GridSpace = 8.1e9;                                % Grid spacing in [Hz] 
P.Fchan     = P.Fc+[-0:0]*P.GridSpace;              % Channel frequency array (number of channels = length(P.Fchan)
P.Channels  = numel(P.Fchan);                       % Number of channels (default 1)
P.Ns          = 4;                                  % Oversampling factor
P.Fs        = P.Ns*(max(abs(P.Fchan(:)-P.Fc))+P.Fb);% Sample rate [Hz], Here, this is Ns x maximum frequency component
P.Fibre.Channels = P.Channels;

%% Bits generation
Nt          = 2^17;                                     % Number of transmitted symbols
P.CodeBits  = randi([0,1],2*P.Channels,Nt*log2(P.M));   % 2 polarizations (even and odd rows) and Nt*log2(P.M) bits
P.CodeIntBits = P.CodeBits;                             % No need for interleaving in this case

%% Binary labeling (this only works for square QAM)
P.SymbolMapping         = 'Custom';       	% We will always use a custom mapping. 
P.Labeling              = 'BRGC';        	% Gray coding
if P.M ==16
    P.CustomSymbolMapping = [2,3,1,0,6,7,5,4,14,15,13,12,10,11,9,8]';
end
if P.M == 64
    P.CustomSymbolMapping = [4,5,7,6,2,3,1,0,12,13,15,14,10,11,9,8,28,29,31,30,26,27,25,24,20,21,23,22,18,19,17,16,52,53,55,54,50,51,49,48,60,61,63,62,58,59,57,56,44,45,47,46,42,43,41,40,36,37,39,38,34,35,33,32]';
end
% P.CustomSymbolMapping   = P.CustomSymbolMapping(randperm(length(P.CustomSymbolMapping)));

%% Constallation centroids in the order of labeling, for General QAM Mod./Demod.
if ~isfield(P,'Constellation')
    P.Constellation = get_MQAM_constellation(P.M); % Power normalized constellation
end
P.LabelledConstellation = P.Constellation(P.CustomSymbolMapping+1);

%% Constellation and Labeling
P.X=P.LabelledConstellation;
P.X=[real(P.X),imag(P.X)];

P.Lbin= dec2bin(P.CustomSymbolMapping)-48;
% Find Subconstellations defined by the labeling
P.m = log2(P.M);        % Bits per symbol
P.Ik0 = zeros(P.M/2,P.m);
P.Ik1 = zeros(P.M/2,P.m);
for kk=1:P.m
    pntr0=1;
    pntr1=1;
    for i=1:P.M
        if P.Lbin(i,kk)==0
            %P.Ik0(pntr0,kk)=i;
            P.Ik0(pntr0,kk)=P.CustomSymbolMapping(i)+1;
            pntr0=pntr0+1;
        else
            %P.Ik1(pntr1,kk)=i;
            P.Ik1(pntr1,kk)=P.CustomSymbolMapping(i)+1;
            pntr1=pntr1+1;
        end
    end
end


%% Indices for mapper
hBitToInt = comm.BitToInteger(log2(P.M));
for index = 1:size(P.CodeBits,1)
    P.TransmittedIndices(index,:) = step(hBitToInt,P.CodeIntBits(index,:)');
end

%% Pulse shaping parameters
P.RRCFilter = 1;     	% [OFF, ON]
P.RRCType = 'Ideal';    % [Ideal,  None]
P.RollOff = 0.001;      % [0, 1]

%% Fiber Transmission Parameters
% N.B. Struct P.Fibre is for parameters related to transmission span (fibre, edfa, &c.)
% P.Fibre.verbose = P.verbose;
P.Fibre.Nspans = 0;         	% Number of transmission spans
P.Fibre.GPU=0;                  % Set to ON to use CUDA for NLSE (default fallback is CPU)
P.Fibre.Length = 0;             % fibre length (km)
P.Fibre.dz = 1;                 % simulation step size (km)
P.Fibre.FiberType = 'ULL';      % Fiber type
P.Fibre.RefWavelength = 1550;   % Reference wavelength (nm)
switch P.Fibre.FiberType
    case 'SMF'
        P.Fibre.Att = 0.2;      % Fibre attenuation (dB/km)
        P.Fibre.AttPump = 0.24; % Fibre attenuation at pump wavelength 1455 nm (dB/km) 
        P.Fibre.D = 17;         % Dispersion parameter at reference wavelength (ps/nm/km)
        P.Fibre.Gamma = 1.2;    % nonlinear parameter (/W/km)
    case 'ULL'
        P.Fibre.Att = 0.165;     % Fibre attenuation (dB/km)
        P.Fibre.AttPump = 0.20; % Fibre attenuation at pump wavelength 1455 nm (dB/km)
        P.Fibre.D = 16.5;       % Dispersion parameter at reference wavelength (ps/nm/km)
        P.Fibre.Gamma = 1.2;    % nonlinear parameter (/W/km)
    case 'Custom'
        P.Fibre.Att = 0.2;      % Fibre attenuation (dB/km)
        P.Fibre.D = 17;         % Dispersion parameter at reference wavelength (ps/nm/km)
        P.Fibre.Gamma = 1.27;   % nonlinear parameter (/W/km)
    otherwise
        error(sprintf(['Unknown fiber type:\n P.FibreType=''' (P.FibreType) ''''])) %#ok<SPERR>
end


P.Fibre.Fb = P.Fb; %redundant parameters, copied into Fibre structure
P.Fibre.Ns = P.Ns;

%% Raman Noise Parameters

P.Fibre.Fc = P.Fc;                               % carrier frequency (Hz)
P.Fibre.Fs = P.Fs;                               % bandwith of the Raman amplifier (Hz) - Should be a function of simulation bandwidth (P.Fs*P.upsample)
  
%% EDFA Parameters
P.Fibre.GdB     = P.Fibre.Length*P.Fibre.Att;   % (Span compensating EDFA) gain
P.Fibre.NFdB    = 7;                            % (Span compensating EDFA) noise figure
% P.Fibre.PsatdBm = 17;                         % (Span compensating EDFA) saturation power

%% DSP Parameters
P.DSP.FullDSP   = 0; % Set to ON to use adaptive equalisers and carrier phase estimation

% Matched Filter
P.DSP.RRCFilter = P.RRCFilter; % Apply a matched filter
P.DSP.RRCType   = P.RRCType;
P.DSP.RollOff   = P.RollOff;

%% BER Estimation

P.BER.SDiscard = 1e3;
P.BER.EDiscard = 1e3;


%% Out
varargout{1} = P;
end

