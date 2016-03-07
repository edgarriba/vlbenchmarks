import localFeatures.*;

mser = VlFeatMser();
%cvc_cnn = cvc_CNN('TORCH_siam2stream_desc_notredame');
cvc_cnn = cvc_CNN('TORCH_iri');

mserWithCNN = DescriptorAdapter(mser, cvc_cnn);

dataset_dir = '/home/eriba/datasets/fountain_dense/urd';
listing = dir(fullfile(dataset_dir, '*.png'));

for i=1:numel(listing)
    fname = fullfile(dataset_dir, listing(i).name);
    [frames descriptors] = mserWithCNN.extractFeatures(fname);

    fileID = fopen(strcat(fname, '.desc'), 'w');
    %fprintf(fileID, '\n');
    
    for i=1:size(descriptors, 2)
        % keypoint
        frame = frames(:, i);
        % x, y, a, b, c
        for j=size(frame):-1:1
            val=frame(j);
            fprintf(fileID, '%i ', frame(j));
        end
        % descriptor
        desc = descriptors(:, i);
        for j=1:size(desc)
            fprintf(fileID, '%i ', desc(j));
        end
        fprintf(fileID, '\n');
    end
end