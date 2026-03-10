%% ========================================================================
% A Deep Learning-based Non-contact Phonocardiogram Measurement Method
% Using mmWave RadarA Deep Learning-based Non-contact Phonocardiogram 
% Measurement Method Using mmWave Radar 
% 
%
% Signal Processing Method for Cardiac Mechanical Activity Extraction
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
%   (With Editor)
% Contents:
%   - Chest localization via cumulative energy evaluation
%   - MDACM-based phase extraction
%   - Micro-motion amplification
%   - Wavelet packet decomposition for subband denoising
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
clc;
clear;
close all;
%% Radar parameter settings 
numADCSamples = 256; 
numADCBits = 16;    
numTX=3;             
numRX = 4;           
numLanes = 2;       
isReal = 0;          
chirpLoop = 1;       
Fs=5e6;                 
c=3*1e8;                
ts=numADCSamples/Fs;
slope=65e12;
B_valid =ts*slope;
detaR=c/(2*B_valid);
startFreq = 77.00e9;
lambda=c/startFreq; 
Tr = 60e-6;
Idle_time = 10e-6;
Tc = Tr+Idle_time;
Bandwidth = slope * Tr ;
BandwidthValid = numADCSamples/Fs*slope;
Range_Res=c/(2*BandwidthValid);
%% Data preprocessing
Filename = 'Data\0\0.bin'; 
Lans = 6;
fid = fopen(Filename,'r');
adcDataRow= fread(fid, 'int16');
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    adcDataRow(adcDataRow > l_max) = adcDataRow(adcDataRow > l_max) - 2^numADCBits;
end
fclose(fid);
process_num = 2000;
fileSize=process_num*chirpLoop*numADCSamples*numTX*numRX*2;  
PRTnum = fix(fileSize/(numADCSamples*numRX));
fileSize = PRTnum * numADCSamples*numRX;
adcData = adcDataRow(1:fileSize);

if isReal
    numChirps = fileSize/numADCSamples/numRX;
    LVDS = zeros(1, fileSize);
    LVDS = reshape(adcData, numADCSamples*numRX, numChirps);
    LVDS = LVDS.';
else
    numChirps = fileSize/2/numADCSamples/numRX; 
    LVDS = zeros(1, fileSize/2);
    counter = 1;
    for i=1:4:fileSize-1
        LVDS(1,counter) = adcData(i) + sqrt(-1)*adcData(i+2);
        LVDS(1,counter+1) = adcData(i+1)+sqrt(-1)*adcData(i+3); 
        counter = counter + 2;
    end
end
%% The radar data is divided into 12 time-division multiplexed (TDM) signals: numTX × numRX.
store_base = zeros(numTX*numRX, numADCSamples*numChirps/numTX);
counter = 1;
r = 1;
adcnum = 1;
is_break = 0;
while(r~=fileSize/2)
    for i = 1:numTX
        for j = 1:numRX
            store_base(counter,adcnum:adcnum+numADCSamples-1) = LVDS(1,r:r+numADCSamples-1);
            counter = counter + 1;
            r = r+numADCSamples;
            if(r==fileSize/2+1)
                is_break = 1;
            end
        end
        if(is_break)
            break;
        end
    end
    if(is_break)
        break;
    end
    counter = 1;
    adcnum =adcnum + numADCSamples;
end
%% Reorganize the data into 12 × ADC × number of chirps transmitted per TX antenna.
processed_adc = zeros(12, numADCSamples, numChirps/numTX);
lopp = 1;
for i = 1:12
    for j = 1:numChirps/numTX
        processed_adc(i,1:256,j)=store_base(i,lopp:lopp+255);
        lopp = lopp+256;
    end
    lopp = 1;
end
%% Reorganize the data (complex data from 4 receive antennas).
LVDS = reshape(LVDS, numADCSamples*numRX, numChirps);
LVDS = LVDS.';
adcData1 = zeros(numRX,numChirps*numADCSamples);
for row = 1:numRX
    for i = 1: numChirps
        adcData1(row, (i-1)*numADCSamples+1:i*numADCSamples) = LVDS(i, (row-1)*numADCSamples+1:row*numADCSamples);
    end
end
%% Phase Extraction
% retVal= reshape(adcData1(1, :), numADCSamples, numChirps);      
data_angle = zeros(12,1999);
processed_adc_max = zeros(12,2000);
for r = 1:12
    retVal = squeeze(processed_adc(r,:,:));
    process_adc=retVal;            
    fft_data = fft(process_adc,numADCSamples); 
    fft_data = fft_data.';
    fft_data_abs = abs(fft_data);
    fft_1d= zeros(numChirps,numADCSamples);
    for j=1:numChirps/numTX
        fft_1d(j,:) = fft_data(j,:);
    end

    [~,Y] = meshgrid((0:numADCSamples-1)*detaR, ...
        (1:numChirps)); 
    % Locate the peak energy point,
    fft_data_last = zeros(1,numADCSamples);
    range_max = 0;
    for j = 1:numADCSamples
        if((j*detaR)<2.5 &&(j*detaR)>0.1)        % The detection range is limited to 0.4–2.5 m.
            for i = 1:numChirps/numTX
                fft_data_last(j) = fft_data_last(j) + fft_data_abs(i,j);
            end
        
            if ( fft_data_last(j) > range_max)
                range_max = fft_data_last(j);
                max_num = j;
            end
        end
    end
    real_data = real(fft_data);
    imag_data = imag(fft_data);


    phi = 0;
    for n = 2:numChirps/numTX-1
        for i = 2:n
            up = real_data(i-1,max_num)*imag_data(i,max_num) - real_data(i,max_num)*imag_data(i-1,max_num);
            phi = up + phi;
        end
        real_x(n) = (lambda*phi)/(4*pi*10e9);
        phi = 0;
    end
    processed_adc_max(r,:) = processed_adc(r,max_num,:);
    figure;
    plot(real_x, LineWidth=1);    
    xlabel('time/points（N）：to each chirp','FontSize',13);
    ylabel('phase','FontSize',13);
    title('phase','FontSize',13);
    grid on;
    data_angle(r,:) = real_x;
end
%% spectrum analysis
data_denoised = zeros(12,1999);
SNR = zeros(1,12);
noise_power = zeros(1,12);
for i = 1:12
    data_denoised(i,:) = wden(data_angle(i,:),'minimaxi','h','sln',4,'db4');
%     figure;plot(data_denoised(i,:));
    Y = fft(data_denoised(i,:));    
    N = length(data_denoised(i,:));   
    P2 = abs(Y/N);              
    P1 = P2(1:floor(N/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = 200*(0:floor(N/2))/N;
%     figure;
%     plot(f, P1);
%     xlabel('frequency (Hz)');
    grid on;
end
%% Cardiac signal extractioan
X = zeros(12,1990);
for i = 1:12
    amplifiedSignal = microMotionAmplification(data_denoised(i,:), 1/200);
    figure;plot(amplifiedSignal(1,1:1990),LineWidth=1);grid on;
    fs = 200;
    N = 50;
    beta = 90;
    Wstop1 = 1 / (fs/2);
    Wstop2 = 2 / (fs/2);
    b = fir1(N, [Wstop1, Wstop2], 'bandpass', kaiser(N+1, beta));

    f_sig = filter(b,1,amplifiedSignal(1,1:1990));
    dc = mean(f_sig);
    f_sig = f_sig - dc;
%     figure;plot(f_sig);
    t = 0:(1/200):(1990-1) * (1/200);
    [sst,f] = wsst(f_sig,fs);
    cor = abs(sst);
%     figure;
%     pcolor(t,f,cor)
%     clim([0, 0.003]); 
%     shading interp
%     xlabel('Seconds')
%     ylabel('Frequency (Hz)')
%     title('Synchrosqueezed Transform')

    wpt = wpdec(f_sig, 3, 'db4');
    %plot(wpt);
    
    X(i,:) = wprcoef(wpt,[3 2]) + wprcoef(wpt,[3 3]) + wprcoef(wpt,[3 4]) + wprcoef(wpt,[3 5]) + wprcoef(wpt,[3 6]);
%     X(i,:) = wprcoef(wpt,[1 1]);
    figure;
    plot(X(i,:));
    grid on;
end
X_mean = mean(X, 1);
figure;plot(X_mean);grid on;




