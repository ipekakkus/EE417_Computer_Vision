%% Stereo Depth Estimation – Classical Pipeline with Full Analysis
% Author: <Baris Bakirdoven>, 2025-04-20
% Enhanced with comprehensive evaluation metrics and visualizations

clear; clc; close all

%% 0. Configuration
leftImgFile   = '000051_10_left.png';     % Left rectified image
rightImgFile  = '000051_10_right.png';    % Right rectified image
calibFile     = '000051_calib.txt';       % Camera calibration
gtDispFile    = '000051_10.png';          % Ground truth disparity (16-bit PNG)

DISPARITY_RANGE = [0 128];                % SGM search range [min max] px
MEDFILT_SIZE = [3 3];                     % Median filter kernel size
UNIQUENESS_THRESH = 15;                   % SGM uniqueness threshold

%% 1. Data Preparation
% Load stereo pair and calibration data
[imgL, imgR, K_left, baseline, stereoParams] = loadAndCalibrateData(...
    leftImgFile, rightImgFile, calibFile);

% Even though KITTI images are pre-rectified, we generate rectifications for future visualization,
% not in our main pipeline
% matrices explicitly for MATLAB's reconstructScene to work
[imgLr, imgRr, stereoParamsRect] = rectifyStereoImages(imgL, imgR, stereoParams, ...
    'OutputView', 'full');  % Use 'full' to maintain original image size

[dMap_R, dMapFilt_R] = computeDisparityMap(imgLr, imgRr, DISPARITY_RANGE, ...
    MEDFILT_SIZE, UNIQUENESS_THRESH);

%% 2. Compute Disparity 
% Compute initial disparity map using Semi-Global Matching
[dMap, dMapFilt] = computeDisparityMap(imgL, imgR, DISPARITY_RANGE, ...
    MEDFILT_SIZE, UNIQUENESS_THRESH);

%% 3. Depth Conversion 
[f, Z] = convertToDepth(dMap, K_left, baseline);  % Raw depth
[~, Z_filt] = convertToDepth(dMapFilt, K_left, baseline); % Filtered depth

%%  4. Load Ground Truth 
gtDisp = loadGroundTruth(gtDispFile);      % Load and scale GT disparity
validGT = gtDisp > 0 & dMapFilt > 0;       % Valid pixels for evaluation

%% 5. Core Visualizations 
%––– Stereo Anaglyph –––
A = stereoAnaglyph(imgL, imgR);
figure('Name','Stereo Anaglyph','NumberTitle','off', ...
       'Position',[200 200 800 600]);
imshow(A);
title('Stereo Anaglyph','FontSize',14);

%––– Filtered Disparity –––
figure('Name','Filtered Disparity','NumberTitle','off', ...
       'Position',[300 200 800 600]);
imagesc(dMapFilt);
colormap(jet);
colorbar('FontSize',12);
axis image off;
title('Filtered Disparity (px)','FontSize',14);

figure('Name','Disparity','NumberTitle','off', ...
       'Position',[300 200 800 600]);
imagesc(dMap);
colormap(jet);
colorbar('FontSize',12);
axis image off;
title('Disparity (px)','FontSize',14);

%––– Filtered Depth –––
figure('Name','Filtered Depth','NumberTitle','off', ...
       'Position',[400 200 800 600]);
imagesc(Z_filt);
colormap(jet);
colorbar('FontSize',12);
axis image off;
title('Filtered Depth (m)','FontSize',14);

figure('Name','Depth','NumberTitle','off', ...
       'Position',[400 200 800 600]);
imagesc(Z);
colormap(jet);
colorbar('FontSize',12);
axis image off;
title('Depth (m)','FontSize',14);

%% 6. Quantitative Evaluation 
% Compute KITTI D1-all metric
[d1all, absErr, badPixels] = computeKittiMetrics(dMapFilt, gtDisp, validGT);
fprintf('KITTI D1-all error = %.2f%%\n', d1all);

% Additional metrics
mae = mean(absErr, 'omitnan');
rmse = sqrt(mean(absErr.^2, 'omitnan'));
fprintf('MAE: %.2f px | RMSE: %.2f px\n', mae, rmse);

%% 7. Error Visualization 
visualizeErrors(absErr, badPixels, imgL, gtDisp, dMapFilt, validGT);

%%  8. Disparity Analysis 
analyzeDisparity(dMapFilt, gtDisp, validGT);

%% 9. Depth Error Analysis 
analyzeDepthError(Z_filt, gtDisp, f, baseline, validGT);


%% 10. 3D Visualization 
visualize3DScene(dMap_R, stereoParams, imgLr);

%% 11. Runtime Analysis 
profileRuntime(imgL, imgR, DISPARITY_RANGE, MEDFILT_SIZE, UNIQUENESS_THRESH);

%% HELPER FUNCTIONS 
function vec = findCalibLine(keyword, fileName)
%FINDCALIBLINE  Return numeric vector after a keyword in KITTI calib files
%
%   vec = FINDCALIBLINE('P_rect_02', 'calib_cam_to_cam.txt');
%
%   Looks for a line that starts with "keyword:" and converts the rest of
%   the line into a numeric row vector.

fid = fopen(fileName,'r');
if fid == -1
    error('Cannot open %s', fileName);
end

vec = [];
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if startsWith(line, [keyword ':'])
        nums = sscanf(line(numel(keyword)+2:end), '%f');  % skip "keyword:"
        vec  = nums(:)';                                  % row vector
        break;
    end
end
fclose(fid);

if isempty(vec)
    error('Keyword "%s" not found in %s', keyword, fileName);
end
end


function [imgL, imgR, K_left, baseline, stereoParams] = loadAndCalibrateData(...
    leftFile, rightFile, calibFile)
    % Load images and parse KITTI calibration
    imgL = imread(leftFile);
    imgR = imread(rightFile);
    
    % Parse calibration (from original code)
    P2_row = findCalibLine('P_rect_02', calibFile);
    P3_row = findCalibLine('P_rect_03', calibFile);
    P2 = reshape(P2_row, 4, 3)';
    P3 = reshape(P3_row, 4, 3)';
    
    K_left = P2(1:3,1:3);
    K_right = P3(1:3,1:3);
    f = K_left(1,1);
    baseline = abs(P2(1,4) - P3(1,4)) / f;
    
    % Build stereo parameters (from original code)
    cameraParamsL = cameraParameters('IntrinsicMatrix', K_left', ...
        'RadialDistortion',[0 0 0], 'TangentialDistortion',[0 0]);
    cameraParamsR = cameraParameters('IntrinsicMatrix', K_right', ...
        'RadialDistortion',[0 0 0], 'TangentialDistortion',[0 0]);
    stereoParams = stereoParameters(cameraParamsL, cameraParamsR, eye(3), ...
        [baseline 0 0]);
end

function [dMap, dMapFilt] = computeDisparityMap(imgL, imgR, dispRange, ...
    filtSize, uniquenessThresh)
    % Compute SGM disparity and apply median filtering
    dMap = disparitySGM(rgb2gray(imgL), rgb2gray(imgR), ...
        'DisparityRange', dispRange, ...
        'UniquenessThreshold', uniquenessThresh);
    dMapFilt = medfilt2(dMap, filtSize, 'symmetric');
end

function [f, Z] = convertToDepth(dMap, K, baseline)
    % Convert disparity map to depth map
    f = K(1,1);
    Z = f * baseline ./ dMap;
    Z(dMap <= 0) = NaN;
end

function gtDisp = loadGroundTruth(gtFile)
    % Load and scale KITTI ground truth disparity
    gtRaw = imread(gtFile);
    gtDisp = double(gtRaw)./256;  % Convert from uint16 to px values
end

function [d1all, absErr, badPixels] = computeKittiMetrics(dMap, gtDisp, validMask)
    % Compute KITTI D1-all metric and error statistics
    absErr = abs(dMap(validMask) - gtDisp(validMask));
    badPixels = absErr > 3 & absErr./gtDisp(validMask) > 0.05;
    d1all = 100 * sum(badPixels) / nnz(validMask);
end

function visualizeErrors(absErr, badPixels, imgL, gtDisp, estDisp, validGT)
    % Create comprehensive error visualizations
    figure('Name','Error Analysis', 'Position',[100 100 1200 400])
    
    % Error map
    subplot(131)
    imagesc(absErr, [0 10])
    colorbar; axis image off
    title('Absolute Disparity Error (px)')
    
    % Error distribution
    subplot(132)
    histogram(absErr, 'BinWidth',0.5, 'EdgeColor','none')
    xlabel('Disparity Error (px)'), ylabel('Frequency')
    title('Error Distribution')
    grid on
    
    % Error overlay
    subplot(133)
    imshow(imgL)
    hold on
    scatter(badPixels(:), imgL, 10, 'r', 'filled', 'MarkerEdgeColor','none')
    alpha(0.3)
    title('Error Regions Overlay (Red)')
end

function analyzeDisparity(estDisp, gtDisp, validMask)
    % Analyze disparity distribution and accuracy curve
    figure('Name','Disparity Analysis', 'Position',[100 100 800 400])
    
    % Disparity histograms
    subplot(121)
    histogram(estDisp(validMask), 'BinWidth',1, 'EdgeColor','none')
    hold on
    histogram(gtDisp(validMask), 'BinWidth',1, 'EdgeColor','none')
    xlabel('Disparity (px)'), ylabel('Frequency')
    legend('Estimated','Ground Truth')
    title('Disparity Distribution')
    
    % Accuracy curve
    subplot(122)
    thresh = 1:10;
    acc = arrayfun(@(t) mean(abs(estDisp(validMask)-gtDisp(validMask)) <= t), thresh);
    plot(thresh, 100*acc, 'LineWidth',2)
    xlabel('Error Threshold (px)'), ylabel('Accuracy (%)')
    title('Pixel Accuracy Curve')
    grid on
end

function analyzeDepthError(Z, gtDisp, f, baseline, validMask)
    % Analyze depth error vs distance
    gtDepth = f * baseline ./ gtDisp;
    depthErr = abs(Z - gtDepth);
    
    figure('Name','Depth Error Analysis')
    depthBins = 0:5:50;
    meanErr = zeros(1, numel(depthBins)-1);
    for i = 1:numel(depthBins)-1
        mask = gtDepth >= depthBins(i) & gtDepth < depthBins(i+1) & validMask;
        meanErr(i) = mean(depthErr(mask), 'omitnan');
    end
    
    bar(depthBins(1:end-1)+2.5, meanErr)
    xlabel('Depth (m)'), ylabel('Mean Absolute Error (m)')
    title('Depth Error vs Distance')
end


function visualize3DScene(dMap, stereoParamsRect, imgLr)
    % Generate 3D point cloud using rectified parameters
    xyzPoints = reconstructScene(dMap, stereoParamsRect);
    
    % Convert to MATLAB pointCloud object
    ptCloud = pointCloud(xyzPoints, 'Color', imgLr);
    
    % Visualize
    figure('Name','3D Visualization');
    pcshow(ptCloud, 'VerticalAxis', 'Y', 'VerticalAxisDir', 'down');
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('3D Point Cloud from SGM Disparity');
end


function profileRuntime(imgL, imgR, dispRange, filtSize, uniquenessThresh)
    % Time disparity computation
    tic
    [dMap, dMapFilt] = computeDisparityMap(imgL, imgR, dispRange, ...
        filtSize, uniquenessThresh);
    time = toc;
    
    fprintf('Disparity computation time: %.2fs\n', time)
end
