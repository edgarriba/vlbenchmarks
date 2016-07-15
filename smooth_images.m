import datasets.*;
import localFeatures.*;
import localFeatures.helpers.*;
import benchmarks.helpers.*;

dataset_dir = '/home/eriba/datasets/fountain_dense/urd';
%dataset_dir = '/home/eriba/datasets/herzjesu_dense/urd';

listing = dir(fullfile(dataset_dir, '*.png'));

for i=1:numel(listing)
  
    % setup inpt image name
    fname_in = strcat(dataset_dir, '/', listing(i).name);

    fprintf('Image %d: %s\n', i, fname_in);

    % read img and apply gaussian filter
    I = imread(fname_in);
    
    %G = fspecial('gaussian', [10 10], 10);
    %Ig = imfilter(I, G, 'same');
    I_out = imresize(I, 0.5);
    
    % serialize descriptors
    fname_out = strcat(dataset_dir, '_smooth/', listing(i).name);
    
    % plot results
    subplot(1,2,1);
    imshow(I);
    
    subplot(1,2,2);
    imshow(I_out);
      
     % save image
    imwrite(I_out, fname_out);
end
