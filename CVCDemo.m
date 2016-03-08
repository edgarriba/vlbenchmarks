function repeatabilityDemo2(resultsPath)
% REPEATABILITYDEMO Demonstrates how to run the repatability benchmark
%   REPEATABILITYDEMO() Runs the repeatability demo.
%
%   REPEATABILITYDEMO(RESULTS_PATH) Run the demo and save the results to
%   path RESULTS_PATH.

% Author: Karel Lenc and Andrea Vedaldi

% AUTORIGHTS

close all
clear all

addpath(genpath(pwd))

if nargin < 1, resultsPath = ''; end;

% --------------------------------------------------------------------
% PART 1: Image feature detectors
% --------------------------------------------------------------------

import datasets.*;
import benchmarks.*;
import localFeatures.*;

% A detector repeatability is measured against a benchmark. In this
% case we create an instance of the VGG Affine Testbed (graffity
% sequence).
% 'boat', gives out of memory
% 'trees', gives out of memory
datasets_names = {'bark', 'bikes', 'graf', 'leuven', 'ubc', 'wall', 'boat', 'trees'};
%datasets_names = {'bark'};

% Next, the benchmark is intialised by choosing various
% parameters. The defaults correspond to the seetting in the original
% publication (IJCV05).

repBenchmark = CVCRepeatabilityBenchmark('Mode','Repeatability');

% The feature detector/descriptor code is encapsualted in a corresponding
% class. For example, VLFeatSift() encapslate the SIFT implementation in
% VLFeat.
%
% In addition to wrapping the detector code, each object instance
% contains a specific setting of parameters (for example, the
% cornerness threshold). In order to compare different parameter
% settings, one simply creates multiple instances of these objects.

% Prepare three detectors, the two from PART 1 and a third one that
% detects MSER image features.

%mser = VlFeatMser('Delta', 10);
mser = VlFeatMser();
siftDetector = VlFeatSift();
siftDetector2 = VggDescriptor();
cvc_cnn1 = cvc_CNN('TORCH_LUA_siam2stream_liberty');
cvc_cnn4 = cvc_CNN('TORCH_LUA_2ch2stream_liberty');
cvc_cnn6 = cvc_CNN('CAFFE_alexnet');
cvc_cnn7 = cvc_CNN('TORCH_iri');
cvc_cnn8 = cvc_CNN('CAFFE_VGG16');

% --------------------------------------------------------------------
% PART 3: Detector matching score
% --------------------------------------------------------------------

% The matching score is similar to the repeatability score, but
% involves computing a descriptor. Detectors like SIFT bundle a
% descriptor as well. However, most of them (e.g. MSER) do not have an
% associated descriptor (e.g. MSER). In this case we can bind one of
% our choice by using the DescriptorAdapter class.
%
% In this particular example, the object encapsulating the SIFT
% detector is used as descriptor form MSER.

mserWithSift = DescriptorAdapter(mser, siftDetector);
mserWithSift2 = DescriptorAdapter(mser, siftDetector2);
mserWithCNN1 = DescriptorAdapter(mser, cvc_cnn1);
%mserWithCNN2 = DescriptorAdapter(mser, cvc_cnn2);
%mserWithCNN3 = DescriptorAdapter(mser, cvc_cnn3);
mserWithCNN4 = DescriptorAdapter(mser, cvc_cnn4);
%mserWithCNN5 = DescriptorAdapter(mser, cvc_cnn5);
mserWithCNN6 = DescriptorAdapter(mser, cvc_cnn6);
mserWithCNN7 = DescriptorAdapter(mser, cvc_cnn7);
mserWithCNN8 = DescriptorAdapter(mser, cvc_cnn8);

featExtractors = {mserWithSift, mserWithSift2, mserWithCNN1, mserWithCNN4, mserWithCNN6, mserWithCNN7};
%featExtractors = {mserWithCNN7};

%% First we will compute the descriptors

%for j = 1:numel(datasets_names)
%    char(datasets_names(j))
%    dataset = datasets.VggAffineDataset('Category', char(datasets_names(j)));
%    for d = 1:numel(featExtractors)
%      for i = 2:dataset.NumImages
%        [framesA descriptorsA] = featExtractors{d}.extractFeatures(dataset.getImagePath(1));
%        [framesB descriptorsB] = featExtractors{d}.extractFeatures(dataset.getImagePath(i));
%      end
%    end
%end

%% Second we will run the tests in parallel

% We create a benchmark object and run the tests as before, but in
% this case we request that descriptor-based matched should be tested.

%matchingBenchmark = CVCRepeatabilityBenchmark('Mode', 'MatchingScore');

% for j = 1:numel(datasets_names)
%     char(datasets_names(j))
%     dataset = datasets.VggAffineDataset('Category', char(datasets_names(j)));
%     matchScore = [];
%     auc = [];
%     numMatches = [];
%     for d = 1:numel(featExtractors)
%       for i = 2:dataset.NumImages
%         [matchScore(d,i) auc(d,i) numMatches(d,i)] = ...
%           matchingBenchmark.testFeatureExtractor(featExtractors{d}, ...
%                                     dataset.getTransformation(i), ...
%                                     dataset.getImagePath(1), ...
%                                     dataset.getImagePath(i));
%       end
%     end
% end

repBenchmark = CVCRepeatabilityBenchmark('Mode','ThresholdBased'); % Repeatability');

for j = 1:numel(datasets_names)

    char(datasets_names(j))
    dataset = datasets.VggAffineDataset('Category', char(datasets_names(j)));

    for d = 1:numel(featExtractors)
      for i = 2:dataset.NumImages
        descriptorMatches = {};
        geometryMatches = {};
        reprojFrames = {};
        [descriptorMatches{d,i}, geometryMatches{d,i}, reprojFrames{d,i}] = ...
            repBenchmark.testFeatureExtractor(featExtractors{d}, ...
                                    dataset.getTransformation(i), ...
                                    dataset.getImagePath(1), ...
                                    dataset.getImagePath(i));
      end
    end
end

%
% %printScores(detectorNames, matchScore*100, 'Match Score');
% %printScores(detectorNames, auc*100, 'Area Under Curve');
% %printScores(detectorNames, numMatches, 'Number of matches') ;
%
% figure(4); clf;
% subplot(1,3,1);
% plotScores(detectorNames, dataset, matchScore*100,'Matching Score');
% subplot(1,3,2);
% plotScores(detectorNames, dataset, auc*100,'Area Under Curve');
% subplot(1,3,3);
% plotScores(detectorNames, dataset, numMatches,'Number of matches');

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
  fprintf('\n');
  for k = 1:numDetectors
    fprintf(formatString,detectorNames{k});
    for l = 2:size(scores,2)
      fprintf('\t%6s',sprintf('%.2f',scores(k,l)));
    end
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
  title(titleText);
  set(gca,'xtick',1:size(score,2));
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
