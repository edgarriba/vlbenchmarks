function storePatches(obj, imagePath, descriptors, featureType, img, frames)

    [framesFile, tmpDir, name, ext] = localFeatures.helpers.getFramesFile(...
        imagePath, featureType);

    if (exist(framesFile, 'file') == 2)
        %obj.debug('Already computed normalized frames');
        return;
    end 
    if (~(exist(tmpDir, 'dir') == 7)), mkdir(tmpDir); end
    
    fileID = fopen(framesFile, 'w');
    % convert descriptor to squared patches
    for i=1:size(descriptors, 2)
        % we assume that patches are squared
        N = sqrt(size(descriptors(:,i), 1));
        patch  = reshape(descriptors(:,i), N, N);
        fname = strcat(strcat(name, '_', int2str(i)), ext);
        fname = fullfile(tmpDir, fname);
        imwrite(patch, fname);
        fprintf(fileID,'%s\n', fname);
        
        % VISUAL DEBUG
%         figure(2);
%         imshow(img);
%         h1 = vl_plotframe(frames(:,i));
%         
%         figure(3);
%         imshow(patch, []);
    end
end