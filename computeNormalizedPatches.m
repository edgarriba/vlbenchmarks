% Compute normalized patches and  save them in disk.
% Return the path of the file which includes a list where
% the patches are saved.
function [frames] = computeNormalizedPatches(imagePath, frames)
    % setup output path
    [pathstr, name, ext] = fileparts(imagePath);
    tmpDir = fullfile(pwd, pathstr, 'tmp');
    framesFile = fullfile(tmpDir, strcat(name, '_list.txt'));

    if (~(exist(tmpDir, 'dir') == 7)), mkdir(tmpDir); end 
    % extract rectified frames
    image = imread(imagePath);
    if(size(image,3)>1), image = rgb2gray(image); end
    image = im2single(image); % If not already in uint8, then convert

    startTime = tic;
    [frames descriptors] = vl_covdet(image, ...
                                   'Frames', frames, ...
                                   'Descriptor', 'Patch', ...
                                   'PatchResolution', 31, ...
                                   'EstimateAffineShape', true, ...
                                   'EstimateOrientation', true, ...
                                   'Verbose');

    fileID = fopen(framesFile, 'w');
    % convert descriptor to squared patches
    for i=1:size(descriptors, 2)
        % we assume that patches are squared
        N = sqrt(size(descriptors(:,i), 1));
        patch = reshape(descriptors(:,i), N, N);           
        fname = strcat(strcat(name, '_', int2str(i)), ext);
        fname = fullfile(tmpDir, fname);
        imwrite(patch, fname);
        fprintf(fileID,'%s\n', fname);
    end
    timeElapsed = toc(startTime);
end