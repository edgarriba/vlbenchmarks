function [patches] = loadPatches(obj, imagePath, featureType)

    [framesFile, tmpDir, name, ext] = localFeatures.helpers.getFramesFile(...
        imagePath, featureType);

    fileID = fopen(framesFile, 'r');
    tline = fgetl(fileID);
    
    i = 1;
    while ischar(tline)
        patch = imread(tline);
        if(size(patch,3)>1), patch = rgb2gray(patch); end
        
        patches(:,:,i) = patch;
        tline = fgetl(fileID);
        i = i + 1;
    end

end