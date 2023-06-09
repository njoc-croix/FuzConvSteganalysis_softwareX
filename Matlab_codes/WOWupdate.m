clc;
clear all;

% Set the input and output folders
inputFolder = 'C:\Users\JEANDELACROIXNTIVUGU\Desktop\OKAY\Cover_Images\';
modificationMapsFolder = 'C:\Users\JEANDELACROIXNTIVUGU\Desktop\OKAY\Modification_Maps\';
fuzzyCorrelationMapsFolder = 'C:\Users\JEANDELACROIXNTIVUGU\Desktop\OKAY\Fuzzy_Correlation_Maps\';
 
% Get a list of image files in the input folder
imageFiles = dir(fullfile(inputFolder, '*.pgm'));
for i = 1:numel(imageFiles)
    inputFile = fullfile(inputFolder, imageFiles(i).name);
    Input_CovI = imread(inputFile);
    % First embedding
    % set the payload capacity
     Payload_Capacity = 0.2;

    % set params
    parameters.p = -1; % holder norm parameter

    fprintf('Calling "My_WOW_main_function" to embed the data for the 1st stego,');
    TACstart = tic;

    %% Run embedding simulation
    [altered_image, distorted] = My_WOW_main_function(Input_CovI, Payload_Capacity, parameters);
    TACend = toc(TACstart);
    fprintf(' ..... embedding successfully done!\n');
    %second embedding 
    I = reshape (altered_image,256,256);
    % set the payload capacity
    Payload_Capacity = 0.2;

    % set parameters
    parameters.p = -1; % holder norm parameter

    fprintf('Calling "My_WOW_main_function" to embed the data for the 2nd stego,');
    TACstart = tic;

    % Run embedding simulation
    [altered_image2, distorted2] = My_WOW_main_function(I, Payload_Capacity, parameters);

    TACend = toc(TACstart);
    fprintf(' ..... embedding successfully done!\n');

    Stego1 = reshape (altered_image,256,256);
    Stego2 = reshape (altered_image2,256,256);
    
    % Computing for modification maps
    Modification_maps = (double(Stego1) - double(Stego2));

    fprintf('\n\nEmbedding done in %.2f seconds, whole image modification rate: %.4f, pixel alteration rate: %.6f\n', TACend, sum(Input_CovI(:) ~= altered_image(:)) / numel(Input_CovI), distorted / numel(Input_CovI));

%figure; imshow(Input_CovI);
%title('Input Image (cover Image)');
figure; imshow(Modification_maps);
title('Modification maps');
saveas(gcf, fullfile(modificationMapsFolder, [num2str(i),'_Modification_Maps', '.png']));
%************************************** 1st stego existence checking******   
%############################################################################################################################
%...................... PART2: fuzzy correlation maps computation.........................

%The fuzzy correlation map is computed using four fundamental input variables: the covariance map, the compass mean, the distance vector matrix, and the pixel intensity matrix (See Fig. 3).
%	covariance map: Covariance maps reveal variable changes and model data dependencies, regardless of physical interpretation, in images' statistical characteristics. 
%	Compass mean: The compass-mean functions like a compass at the top of a map, guiding the orientation of surrounding pixels. It calculates correlations between neighboring pixels and the central pixel, producing average intensities.
%	Distance vector matrix: The distance vector matrix consists of scalar values that represent the distance between a central pixel and any pixel located at position (i,j).
%	Pixel intensity matrix: Based on the presented modification map, the arrangement of pixels within it reveals the pixel intensity matrix. 

%................Input membership functions calculations...................

%-------------- 1)Covariance maps calculation from the modification maps---   
    imgfmaps = double(Modification_maps);

% Convert the image to double precision
    imgfmaps = im2double(imgfmaps);
% Compute the mean of the image
    img_mean = mean(imgfmaps, 'all');
% Subtract the mean from each pixel
    img_sub = imgfmaps - img_mean;
% Compute the covariance matrix of the image
    covariance_map = reshape(cov(img_sub),256,256); 
% Display the covariance matrix & Ploting the resuted image
%disp('Covariance matrix:');
%disp(covariance_map);

%------ 2) Pixel intensity matrix calculation from the modification maps---

    Pixel_int_m = double(Modification_maps);

% Get the size of the modification maps
    [rows, cols, ~] = size(Pixel_int_m);

% Initialize an array to store pixel intensities
    pixel_intensities_1 = zeros(rows, cols);

% Iterate over each pixel and store its intensity
    for row = 1:rows
        for col = 1:cols
            pixel_intensities_1(row, col) = Pixel_int_m(row, col);
        end
    end
    pixel_intensities = reshape ( pixel_intensities_1,256,256);
% Display the pixel intensity array
%disp(pixel_intensities);

%------ 3) Distance vector matrix calculation from the modification maps---

% Read in the image and convert to double datatype
    imgdist_vec_M = double(Modification_maps);
% Get the dimensions of the image
    [h, w] = size(imgdist_vec_M);

% Compute the center of the image
    center_x = floor(w / 2) + 1;
    center_y = floor(h / 2) + 1;

% Create a grid of x and y values
    [x, y] = meshgrid(1:w, 1:h);

% Compute the distance matrix from the center of the image

    distance_vector_1 = sqrt((x - center_x) .^ 2 + (y - center_y) .^ 2);
    distance_vector = reshape (distance_vector_1,256,256);
%disp(distance_vector);

%------- 4)Compass mean calculation from the modification maps-----------
% Load the modification maps
    img = double(Modification_maps);
% Initialize the compass kernel
    compass_kernel = [-1,-1,-1; -1,0,-1; -1,-1,-1];

% Set the size of the neighborhood
    neighborhood_size = 3;

% Pad the image with zeros to handle the edges
    padded_img = padarray(img, [floor(neighborhood_size/2) floor(neighborhood_size/2)]);

% Initialize the output image
    output_img = zeros(size(img));

% Compute the compass mean for each pixel in the image
    for i = 1:size(img,1)
        for j = 1:size(img,2)
        
        % Extract the neighborhood of the current pixel
            neighborhood = padded_img(i:i+neighborhood_size-1, j:j+neighborhood_size-1);
        
        % Compute the compass mean
            correlation = sum(sum(neighborhood .* compass_kernel));
            if correlation >= 0
               output_img(i,j) = sum(sum(neighborhood)) / (neighborhood_size^2);
            else
               output_img(i,j) = img(i,j);
            
            end
        end
    end
    compass_mean = reshape (uint8(output_img),256,256);
% Display the compass mean matrix
%disp(compass_mean);
                   %######################################################################################
% Create a Fuzzy Inference System (FIS) for fuzzy correlation maps, TFC1_system

    TFC1_system = newfis('fuzzycorrelationmaps');

%Specify  the inputs of TFC1_system.

    TFC1_system = addvar(TFC1_system,'input','covariance_map',[-1 1]);
    TFC1_system = addvar(TFC1_system,'input','pixel_intensities',[0 255]);
    TFC1_system = addvar(TFC1_system,'input','distance_vector',[0 255]);
    TFC1_system = addvar(TFC1_system,'input','compass_mean',[0 2]);

%Specify Gaussian membership functions for each input.

%Specify Gaussian membership functions for for Input1 "covariance_map"

    nx = -0.0001; ny = 0.0005;
    TFC1_system = addmf(TFC1_system,'input',1,'Low','gaussmf',[nx 0.01]);
    TFC1_system = addmf(TFC1_system,'input',1,'medium','gaussmf',[0.01 ny]);
    TFC1_system = addmf(TFC1_system,'input',1,'high','gaussmf',[ny 1]);

%Specify Gaussian membership functions for for Input2 "pixel_intensities"

    ax = 0.5; ay = 1.5;
    TFC1_system = addmf(TFC1_system,'input',2,'Dark','gaussmf',[-1 ax]);
    TFC1_system = addmf(TFC1_system,'input',2,'Average','gaussmf',[ax ay]);
    TFC1_system = addmf(TFC1_system,'input',2,'Clear','gaussmf',[ay 255]);

%Specify Gaussian membership functions for for Input3 "distance_vector"

    ux =75 ; uy = 150;
    TFC1_system = addmf(TFC1_system,'input',3,'Close','gaussmf',[-1 ux]);
    TFC1_system = addmf(TFC1_system,'input',3,'Medium','gaussmf',[ux uy]);
    TFC1_system = addmf(TFC1_system,'input',3,'Far','gaussmf',[uy 255]);

%Specify Gaussian membership functions for for Input 4 "compass_mean"

    nx = -0.05; ny = 1;
    TFC1_system = addmf(TFC1_system,'input',4,'Dark','gaussmf',[nx 1]);
    TFC1_system = addmf(TFC1_system,'input',4,'Average','gaussmf',[1 ny]);
    TFC1_system = addmf(TFC1_system,'input',4,'Clear','gaussmf',[ny 2]);

%Specify the intensity of the fuzzy_correlation_maps as an output of TFC1_system.

    TFC1_system = addvar(TFC1_system,'output','fuzzy_correlation_maps',[0 255]);

%Specify the triangular membership functions for fuzzy_correlation_maps.

    TFC1_system = addmf(TFC1_system,'output',1,'Low','trimf',[0 64 127]);
    TFC1_system = addmf(TFC1_system,'output',1,'Medium','trimf',[64 127 191]);
    TFC1_system = addmf(TFC1_system,'output',1,'High','trimf',[127 191 255]);


%Add rules 

    r1 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r2 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Low)';
    r3 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Low)';
    r4 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r5 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Low)';
    r6 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r7 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r8 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Low)';
    r9 = 'If (covariance_map is Low) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r10 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r11 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Low)';
    r12 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r13 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r14 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r15 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r16 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r17 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r18 = 'If (covariance_map is Low) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r19 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r20 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r21 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r22 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r23 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r24 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r25 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r26 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r27 = 'If (covariance_map is Low) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r28 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r29 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Low)';
    r30 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r31 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r32 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r33 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r34 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r35 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r36 = 'If (covariance_map is medium) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r37 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r38 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r39 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r40 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r41 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is High)';
    r42 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r43 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r44 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is High)';
    r45 = 'If (covariance_map is medium) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r46 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r47 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r48 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r49 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r50 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r51 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r52 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r53 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is High)';
    r54 = 'If (covariance_map is medium) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r55 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r56 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Low)';
    r57 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r58 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r59 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r60 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r61 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r62 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r63 = 'If (covariance_map is high) and (compass_mean is Dark) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r64 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Low)';
    r65 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r66 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is Medium)';
    r67 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r68 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r69 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r70 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r71 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r72 = 'If (covariance_map is high) and (compass_mean is Average) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r73 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r74 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is Average) then (fuzzy_correlation_maps is Medium)';
    r75 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Far) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r76 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is Medium)';
    r77 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is Average) then (fuzzy_correlation_maps is High)';
    r78 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Medium) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';
    r79 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is Dark) then (fuzzy_correlation_maps is High)';
    r80 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is Average) then (fuzzy_correlation_maps is High)';
    r81 = 'If (covariance_map is high) and (compass_mean is Clear) and (distance_vector is Close) and (pixel_intensities is clear) then (fuzzy_correlation_maps is High)';


    r = char(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30,r31,r32,r33,r34,r35,r36,r37,r38,r39,r40,r41,r42,r43,r44,r45,r46,r47,r48,r49,r50,r51,r52,r53,r54,r55,r56,r57,r58,r59,r60,r61,r62,r63,r64,r65,r66,r67,r68,r69,r70,r71,r72,r73,r74,r75,r76,r77,r78,r79,r80,r81);

    TFC1_system = parsrule(TFC1_system, r); % Add the rules to the FIS object
    showrule(TFC1_system); % Show the updated rules in the FIS object

% Set the output directory and filename
% filename = '12.pgm';
% output_path = fullfile(output_dir, filename);

% Create the correlation maps

    Ieval = zeros(size(I)); % Preallocate the output matrix
    for ii = 1:size(I,1)
        input_data = [covariance_map(ii,:); compass_mean(ii,:); distance_vector(ii,:); pixel_intensities(ii,:)];
        input_data = double(input_data);
        Ieval(ii,:) = evalfis(input_data, TFC1_system);
    end
        figure('Name', 'Ieval');
        image(Ieval, 'CDataMapping', 'scaled'); 
        colormap('default'); 
        title('Fuzzy correlation maps');
        % Save the fuzzy correlation maps in the fuzzyCorrelationMapsFolder
    
    fuzzyCorrelationMapsFolder2 = 'C:\Users\JEANDELACROIXNTIVUGU\Desktop\OKAY\Fuzzy_Correlation_Maps\';
    ModimageFiles = dir(fullfile(modificationMapsFolder, '*.png'));
    
    for k = 1283:numel(ModimageFiles)
        [~, name, ~] = fileparts(imageFiles(k).name);
        fuzzyCorrelationMapFileName = fullfile(fuzzyCorrelationMapsFolder2, [num2str(k),'_Fuzzy_Correlation_Maps','.png']);
        saveas(gcf, fuzzyCorrelationMapFileName);
    end 

end
%#######################################################################

% END
disp('END OF THE FUZZY CORRELATION MAPS COMPUTATION');
