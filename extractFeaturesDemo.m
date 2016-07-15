import datasets.*;
import localFeatures.*;
import localFeatures.helpers.*;
import benchmarks.helpers.*;

% use to recompile vl_feat

detector_names = {'hessian', 'harris', 'dog'};
detector_names = {'hessian'};

descriptor_names = {'iri', 'tfeat', 'sift'};
descriptor_names = {'sift'};

dataset_dir = '/home/eriba/datasets/fountain_dense/urd';
%dataset_dir = '/home/eriba/datasets/herzjesu_dense/urd';

listing = dir(fullfile(dataset_dir, '*.png'));

% setup descriptor

for k=1:numel(detector_names)
    
    DET_NAME = detector_names(k);
    
    % setup detector

    if strcmp(DET_NAME, 'harris')
        detector = VggAffine('detector', 'haraff');
    elseif strcmp(DET_NAME, 'dog')
        detector = VlFeatCovdet('Method', 'DoG');
    elseif strcmp(DET_NAME, 'hessian')
        detector = VlFeatCovdet('Method', 'Hessian', ...
                                'Descriptor', 'Patch', ...
                                'PatchResolution', 20, ...
                                'EstimateAffineShape', false, ...
                                'EstimateOrientation', true, ...
                                'PatchRelativeExtent', 6, ...
                                'PatchRelativeSmoothing', 1.2, ...
                                'PeakThreshold', 0.00001, ...
                                'EdgeThreshold', 10, ...
                                'DoubleImage', false);
    else
    end

    for j=1:numel(descriptor_names)

        DESC_NAME = descriptor_names(j);

        if strcmp(DESC_NAME, 'iri')
            descriptor = cvc_CNN('Name', 'IRI_NET');
        elseif strcmp(DESC_NAME, 'tfeat')
            descriptor = cvc_CNN('Name', 'TFeat_MARGIN_STAR_E60');
        elseif strcmp(DESC_NAME, 'sift')
            %descriptor = cvc_VlFeatSift();
            %descriptor = VlFeatSift();
            descriptor = VlFeatCovdet('Descriptor', 'SIFT', ...
                                      'PatchResolution', 15, ...
                                      'PatchRelativeExtent', 7.5, ...
                                      'PatchRelativeSmoothing', 1, ...
                                      'DoubleImage', false);

        else
        end

        featExtractor = DescriptorAdapter(detector, descriptor);

        for i=1:numel(listing)

            % setup inpt image name
            fname_in = strcat(dataset_dir, '/', listing(i).name);
            
            fprintf('Image %s\n', fname_in);

            % compute descriptor
            [frames descriptors] = featExtractor.extractFeatures(fname_in);
%             [frames descriptors] = featExtractor.FrameDetector.extractFeatures(fname_in);

            % serialize descriptors
            fname_out = strcat(dataset_dir, '/descriptors/', listing(i).name);

            OUT_NAME = strcat(fname_out, '.', DET_NAME, '.', DESC_NAME);
            writeFeatures(char(OUT_NAME), frames, descriptors);

        end
    end
end
