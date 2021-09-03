function varargout = SItoOpt( flag, varargin )
% Converts SI to optical units optical to be consistent with industry units.
%
%
% INPUTS:
% flag - determines the conversion way
%        1 --> SI to Opt
%       -1 --> Opt to SI
%
% P - parameter structure
%
% RETURNS:
% P - parameter structure
%
% Author: Sezer Erkilinc, Nov 2013.
% Modified: Gabriele Liga, Dec 2013

if nargin == 0
    error('Conversion type is not specified. Parameters for conversion not found !!!');
elseif nargin == 1
    if flag == 1
        disp('Conversion type set to SI to optical.');
    elseif flag == -1
        disp('Conversion type set to optical to SI.');
    end
    error('Parameters for conversion not found !!!');
else
    P            = varargin{1};
    Pnames       = fieldnames(P);
    N_parameters = length(Pnames);
end

for ii=1:N_parameters,
    
    switch Pnames{ii}
        case 'Att'
            if flag == 1
                P.Att = (P.Att*4.343)/1e-3;  % [Np/m] --> [dB/km]
            elseif flag == -1
                P.Att = (P.Att/4.343)/1e3;   % [dB/km] --> [Np/m]
            end
            
%         case 'Leff'
%             if flag == 1
%                 P.Leff = P.Leff*1e-3;  % [m] --> [km]
%             elseif flag == -1
%                 P.Leff = P.Leff*1e3;  % [km] --> [m]
%             end
            
        case 'Length'
            if flag == 1
                P.Length = P.Length*1e-3;   % [m] --> [km]
            elseif flag == -1
                P.Length = P.Length*1e3;    %  [km] --> [m]
            end
            
        case 'totalFibreSpan'
            if flag == 1
                P.totalFibreSpan = P.totalFibreSpan*1e-3;   % [m] --> [km]
            elseif flag == -1
                P.totalFibreSpan = P.totalFibreSpan*1e3;    %  [km] --> [m]
            end
            
        case 'dz'
            if flag == 1
                P.dz = P.dz*1e-3;   % [m] --> [km]
            elseif flag == -1
                P.dz = P.dz*1e3;    % [km] --> [m]
            end
            
        case 'RefWavelength'
            if flag == 1
                P.RefWavelength = P.RefWavelength*1e9;  % [m] --> [nm]
            elseif flag == -1
                P.RefWavelength = P.RefWavelength*1e-9;  % [nm] --> [m]
            end
            
        case 'D'
            if flag == 1
                P.D = P.D*(1e12/1e9/1e-3);  % [s/m^2] --> [ps/nm/km]
            elseif flag == -1
                P.D = P.D*(1e-12/1e-9/1e3);  %  [ps/nm/km] --> [s/m^2]
            end
            
        case 'S'
            if flag == 1
                P.S = P.S*1e3;  % [s/m^3] --> [ps/nm^2/km]
            elseif flag == -1
                P.S = P.S*1e-3;  % [ps/nm^2/km] --> [s/m^3]
            end
            
        case 'PMD'
            if flag == 1
                P.PMD = P.PMD*1e12*10^(3/2);  % [s/m^0.5] --> [ps/km^0.5]
            elseif flag == -1
                P.PMD = P.PMD*1e-12*10^(-3/2);  % [ps/km^0.5]  --> [s/m^0.5]
            end
            
        case 'GammaBP'
            if flag == 1
                P.GammaBP = P.GammaBP/1e-3;  % [/(W*m)] --> [/(W*km)]
            elseif flag == -1
                P.GammaBP = P.GammaBP/1e3;  % [/(W*km)] --> [/(W*m)]
            end
        case 'Gamma'
            if flag == 1
                P.Gamma = P.Gamma/1e-3;  % [/(W*m)] --> [/(W*km)]
            elseif flag == -1
                P.Gamma = P.Gamma/1e3;  % [/(W*km)] --> [/(W*m)]
            end
            
        otherwise
            % warning('Parameter, %s, is not found!!!', Pnames{ii} );
    end
    
end

if N_parameters == 0,
    warning('No parameter is found !!!');
else
    varargout{1} = P;
end
end