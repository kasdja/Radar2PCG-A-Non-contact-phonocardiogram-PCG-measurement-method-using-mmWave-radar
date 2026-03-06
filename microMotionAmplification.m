%% ========================================================================
% A Deep Learning-based Non-contact Phonocardiogram Measurement Method
% Using mmWave RadarA Deep Learning-based Non-contact Phonocardiogram 
% Measurement Method Using mmWave Radar 
% 
%
% Micro-motion amplification
% ========================================================================
%
% Author:  Haozhe Liu
% Email:   2021117151@stumail.nwu.edu.cn
% Date:    March 2026
%
% Description:
%   This repository contains the MATLAB implementation of the cardiac signal 
%   extraction Method described in our paper:
%
%   Haozhe Liu, et al., "A Deep Learning-based Non-contact Phonocardiogram 
%   Measurement Method Using mmWave RadarA Deep Learning-based Non-contact 
%   Phonocardiogram Measurement Method Using mmWave Radar ," 2026.
%   
%   (Manuscript in Preparation)
% Contents:
%   - Micro-motion amplification
%
% Hardware:
%   - TI IWR1843 mmWave Radar + DCA1000EVM
%   - Eko Core 500 Digital Stethoscope (reference device)
%
% License:
%   This code is released under the MIT License.
%   You are free to use, modify, and distribute this code, provided that
%   proper citation is given to the original paper.
%
% Disclaimer:
%   This code is provided for academic and research purposes only. 
%   It is NOT intended for clinical diagnosis or medical decision-making. 
%   The authors assume no responsibility for any consequences arising 
%   from the use of this code.
%
%   This is a partial release accompanying a manuscript currently under 
%   peer review. The full codebase will be made available upon paper acceptance.
%
% ========================================================================
function amplifiedSignal = microMotionAmplification(inputSignal, h)
[M, T] = size(inputSignal);
kernel = [1, 2, -1, -4, -1, 2, 1];
amplifiedSignal = zeros(M, T);
for mIdx = 1:M
    rowData = inputSignal(mIdx, :);  
    rowFiltered = conv(rowData, kernel, 'same') / (16 * h^2);
    amplifiedSignal(mIdx, :) = rowFiltered;
end
end

