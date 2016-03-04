function MiniProva(resultsPath)
% REPEATABILITYDEMO Demonstrates how to run the repatability benchmark
%   REPEATABILITYDEMO() Runs the repeatability demo.
%
%   REPEATABILITYDEMO(RESULTS_PATH) Run the demo and save the results to
%   path RESULTS_PATH.

% Author: Karel Lenc and Andrea Vedaldi

% AUTORIGHTS

close all
clear all

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

% 'boat', removed since makes vlfeat crash (check image type)
% 'trees', gives out of memory
%datasets_names = {'bark', 'bikes', 'graf', 'leuven', 'ubc', 'wall'};
datasets_names = {'graf'};

dataset = datasets.VggAffineDataset('Category','graf');

% Next, the benchmark is intialised by choosing various
% parameters. The defaults correspond to the seetting in the original
% publication (IJCV05).

repBenchmark = CVCRepeatabilityBenchmark('Mode','ThresholdBased'); % Repeatability');

vlsift = VlFeatSift();
featExtractors = {vlsift};

% Now we are ready to run the repeatability test. We do this by fixing
% a reference image A and looping through other images B in the
% set. To this end we use the following information:
%
% dataset.NumImages:
%    Number of images in the dataset.
%
% dataset.getImagePath(i):
%    Path to the i-th image.
%
% dataset.getTransformation(i):
%    Transformation from the first (reference) image to image i.
%
% Like for the detector output (see PART 1), VLBenchmarks caches the
% output of the test. This can be disabled by calling
% repBenchmark.disableCaching().

repeatability = [];
numCorresp = [];

imageAPath = dataset.getImagePath(1);
for d = 1:numel(featExtractors)
  for i= 4;
 
    [descriptorMatches, geometryMatches, reprojFrames] = ...
      repBenchmark.testFeatureExtractor(featExtractors{d}, ...
                                dataset.getTransformation(i), ...
                                dataset.getImagePath(1), ...
                                dataset.getImagePath(i));
    
    figure;
        
    for j=1:length(descriptorMatches)
    
        res = descriptorMatches{j};

        subplot(2,5,j);
        plot(res.recall,res.precision), xlabel('Recall'), ylabel('Precision');
        title(res.method);
        subplot(2,5,5+j);
        plot(res.recallGM,res.precision), xlabel('RecallGM'), ylabel('Precision');
        title(res.method);        
    end
  end
end


