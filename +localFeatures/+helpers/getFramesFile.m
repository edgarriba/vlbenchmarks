function [framesFile, tmpDir, name, ext] = getFramesFile(imagePath, featureType)

    % setup output path
    [pathstr, name, ext] = fileparts(imagePath);

    % check detector type
    if ~isempty(strfind(featureType,'VLFeatCovDet_DoG'))
        tmpDir = fullfile(pwd, pathstr, 'tmp_dog');
    elseif ~isempty(strfind(featureType,'VLFeatCovDet_Hessian'))
        tmpDir = fullfile(pwd, pathstr, 'tmp_hessian');
    elseif ~isempty(strfind(featureType,'VGG Affine_hesaff'))
        tmpDir = fullfile(pwd, pathstr, 'tmp_hessian_vgg');      
    else
        tmpDir = fullfile(pwd, pathstr, 'tmp');
    end

    framesFile = fullfile(tmpDir, strcat(name, '_list.txt'));
end