function repeatabilityDemo2(resultsPath)
% REPEATABILITYDEMO Demonstrates how to run the repatability benchmark
%   REPEATABILITYDEMO() Runs the repeatability demo.
%
%   REPEATABILITYDEMO(RESULTS_PATH) Run the demo and save the results to
%   path RESULTS_PATH.

% Author: Karel Lenc and Andrea Vedaldi

% AUTORIGHTS

close all
clearvars *

%addpath(genpath(pwd))

if nargin < 1, resultsPath = ''; end;

% --------------------------------------------------------------------
% PART 1: Image feature detectors
% --------------------------------------------------------------------

import datasets.*;
import benchmarks.*;
import localFeatures.*;

exp_name = 'exp3';

% image sequences
datasets_names = {'bark', 'bikes', 'graf', 'leuven', 'ubc', 'wall', 'boat', 'trees'};
datasets_names = {'graf'};

% type descriptors
descriptor_types = { 'SIFT',...
                     'SIFT_COVDET',...
                     'DEEPCOMPARE_siam_liberty',...
                     'DEEPCOMPARE_siam2stream_liberty',...
                     'IRI_NET',...
                     'MATCHNet_liberty',...
                     'TFeat_MARGIN_E60',...
                     'TFeat_MARGIN_STAR_E60',...
                     'TFeat_SOFTMAX_E01',...
                     'TFeat_SOFTMAX_STAR_E01',...
                    };
descriptor_types = { 'SIFT_COVDET',...
                     'DEEPCOMPARE_siam_liberty',...
                     'DEEPCOMPARE_siam2stream_liberty',...
                     'IRI_NET',...
                     'MATCHNet_liberty',...
                     'TFeat_MARGIN_STAR_E60',...
                    };
descriptor_names = {'SIFT', 'DeepCompare', 'DeepCompare-2str', 'DeepDesc', 'MatchNet', 'TFeat'};

% type detectors
detector_names = {'haraff','dog','hessian'};
detector_names = {'dog'};


%% Run the tests

repBenchmark = CVCRepeatabilityBenchmark('Mode','ThresholdBased',...
                                         'maxSamples', 1000,...
                                         'overlapError', 0.5);

for p=1:numel(detector_names)
    
    % initialize keypoints detector
    detector_name = char(detector_names(p));
    if strcmp(detector_name, 'haraff')
        detector = VggAffine('detector', 'haraff');
        label = 'Harris-Affine ';
    elseif strcmp(detector_name, 'dog')
        detector = VlFeatCovdet('Method', 'DoG');
        label = 'DoG ';    
    elseif strcmp(detector_name, 'hessian')
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
         label = 'Hessian ';
    end
    
    for j = 1:numel(datasets_names)
        
        % initialize dataset
        char(datasets_names(j))
        dataset = datasets.VggAffineDataset('Category',  char(datasets_names(j)));
        
        auc = [];
        count = 1;
        detectorNames  = {};

        for m=1:numel(descriptor_types)

            % instantiate descriptor
            if strcmp(descriptor_types(m), 'SIFT')
                descriptor = cvc_VlFeatSift();
            elseif strcmp(descriptor_types(m), 'SIFT_COVDET')
                descriptor = VlFeatCovdet('Descriptor', 'SIFT', ...
                                          'PatchResolution', 15, ...
                                          'PatchRelativeExtent', 7.5, ...
                                          'PatchRelativeSmoothing', 1, ...
                                          'DoubleImage', false);
            else
                descriptor = cvc_CNN('Name', char(descriptor_types(m)));
            end
            
            featExtractor = DescriptorAdapter(detector, descriptor);
            detectorNames{count} = char(strcat(label, ' - ', lower(descriptor_types(m))));
            count = count + 1;

            for i = 2:dataset.NumImages
                [descriptorMatches, geometryMatches, reprojFrames] = ...
                    repBenchmark.testFeatureExtractor(featExtractor, ...
                                                      dataset.getTransformation(i), ...
                                                      dataset.getImagePath(1), ...
                                                      dataset.getImagePath(i));
                    % AUC
                    if numel(descriptorMatches) > 0
                        auc(m,i) = descriptorMatches{1,2}.auc;
                    else
                        auc(m,i) = 0;
                    end
                    
%                     if i==2
%                         figure;
%                         imagePath = dataset.getImagePath(i);
%                         imshow(imread(imagePath));
%                         frames = featExtractor.extractFeatures(imagePath);
%                         hold on ;
%                         vl_plotframe(frames(:,end-500:end)) ;
%                     end

            end
        end
        
        figure;
        plotScores(descriptor_names, dataset, auc*100,'Area Under Curve');
        printScores(descriptor_names, auc*100, 'Area Under Curve');

        % compute mAP and serialize
        auc = auc(:, 2:end);
        mAP = mean(auc, 2);
        out = [auc mAP];

        % serialize results
        name_file = strcat(exp_name, '_', detector_name);
        name_file = strcat(name_file, '_', char(datasets_names(j)));
        fid=fopen(strcat(char(name_file), '.txt'), 'w');
        fprintf(fid, '%f %f %f %f %f %f\n', out(:,:)');

    end
end


%%
% --------------------------------------------------------------------
% Helper functions
% --------------------------------------------------------------------

function printScores(detectorNames, scores, name)
  numDetectors = numel(detectorNames);
  maxNameLen = length('Method name');
  for k = 1:numDetectors
    maxNameLen = max(maxNameLen,length(detectorNames{k}));
  end
  fprintf(['\n', name,':\n']);
  formatString = ['%' sprintf('%d',maxNameLen) 's:'];
  fprintf(formatString,'Method name');
  for k = 2:size(scores,2)
    fprintf('\tImg#%02d',k);
  end
  fprintf('\tmAP',k);
  fprintf('\n');
  for k = 1:numDetectors
    fprintf(formatString,detectorNames{k});
    for l = 2:size(scores,2)
      fprintf('\t%6s',sprintf('%.2f',scores(k,l)));
    end
    fprintf('\t%6s',sprintf('%.2f',mean(scores(k,2:l), 2)));
    fprintf('\n');
  end
end

function plotScores(detectorNames, dataset, score, titleText)
  xstart = max([find(sum(score,1) == 0, 1) + 1 1]);
  xend = size(score,2);
  xLabel = dataset.ImageNamesLabel;
  xTicks = dataset.ImageNames;
  plot(xstart:xend,score(:,xstart:xend)','+-','linewidth', 2); hold on ;
  ylabel(titleText) ;
  xlabel(xLabel);
  set(gca,'XTick',xstart:1:xend);
  set(gca,'XTickLabel',xTicks);
  %title(titleText);
  %set(gca,'xtick',1:size(score,2));
  maxScore = max([max(max(score)) 1]);
  meanEndValue = mean(score(:,xend));
  legendLocation = 'SouthEast';
  if meanEndValue < maxScore/2
    legendLocation = 'NorthEast';
  end
  legend(detectorNames,'Location',legendLocation);
  grid on ;
  maxScore = (maxScore>100).*maxScore + (maxScore<=100).*100;
  axis([xstart xend 0 maxScore]);
end
end
