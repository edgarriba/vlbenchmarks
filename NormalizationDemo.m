function BMVADemo(resultsPath)
% BMVADemo Demonstrates how to run the repatability benchmark
%   BMVADemo() Runs the repeatability demo.
%
%   BMVADemo(RESULTS_PATH) Run the demo and save the results to
%   path RESULTS_PATH.

% Author: Daniel Ponsa. Based on repeatabilityDemo

% AUTORIGHTS

close all

if nargin < 1, resultsPath = ''; end;

% --------------------------------------------------------------------
% PART 1: Image feature detectors
% --------------------------------------------------------------------

import datasets.*;
import benchmarks.*;
import localFeatures.*;

% The feature detector/descriptor code is encapsualted in a corresponding
% class. For example, VLFeatSift() encapslate the SIFT implementation in
% VLFeat.
%
% In addition to wrapping the detector code, each object instance
% contains a specific setting of parameters (for example, the
% cornerness threshold). In order to compare different parameter
% settings, one simply creates multiple instances of these objects.

dataset = datasets.SyntheticDataset('Category','synthGraf');

covdetPatch = localFeatures.VlFeatCovdet('Method','HarrisLaplace',...
                                         'Descriptor','patch');

covdetAffOrPatch = localFeatures.VlFeatCovdet('Method','HarrisLaplace',...
                                'EstimateAffineShape',true,...
                                'EstimateOrientation',true,...
                                'Descriptor','patch'); %'patch');
                                                       
featExtractors = {covdetPatch, covdetAffOrPatch};
                          
                       
repBenchmark = RepeatabilityBenchmark('Mode','MatchingScore');

imgIndex = 5; % Rotacio 90Âº
imgIndex = 2; % Traslacio
imgIndex = 4; % Rotacio 20Âº

score = [];
numCorresp = [];
corresps = [];
reprojFrames = {};

for d = 1:numel(featExtractors)
  for i = 5
    [score(d,i), numCorresp(d,i), corresps{d,i}, reprojFrames{d,i}] = ...
         repBenchmark.testFeatureExtractor(featExtractors{d}, ...
                                dataset.getTransformation(i), ...
                                dataset.getImagePath(1), ...
                                dataset.getImagePath(i));
                            
    % Display of the mached elipses
    imshow(imread(dataset.getImagePath(i)));
    
    benchmarks.helpers.plotFrameMatches(corresps{d,i},reprojFrames{d,i},...
        'IsReferenceImage',false,'PlotMatchLine',false);
                            
    pause(1)
  end
end

% Demostration that Affine and Orientation normalization afects descriptor
% matching.
detectorNames = {'CovdetPatch','CovdetAffOrPatch'};
printScores(detectorNames, score, 'Matching Score');
printScores(detectorNames, numCorresp, 'Number of correspondences');

% --------------------------------------------------------------------
% Display of the descriptor of matched elipses.
% --------------------------------------------------------------------

for d = 1:numel(featExtractors)
  for i = 5
     
    reprojectedFrames = reprojFrames{d,i};
    descriptorMatches = corresps{d,i};
    
    croppedFrames1 = reprojectedFrames{1};
    croppedFrames2 = reprojectedFrames{2};
        
    [croppedFrames1 desc1] = featExtractors{d}.extractDescriptors(dataset.getImagePath(1), croppedFrames1);
    [croppedFrames2 desc2] = featExtractors{d}.extractDescriptors(dataset.getImagePath(i), croppedFrames2);

    side = sqrt(size(desc1,1));
    
    matchedFrame = 1;
    imMatches = [];
    
    for k=1:5
    
        while descriptorMatches(matchedFrame) == 0
            matchedFrame = matchedFrame + 1;
        end
  
        frameDesc1 = reshape(desc1(:,matchedFrame),side,side);
        frameDesc2 = reshape(desc2(:,descriptorMatches(matchedFrame)),side,side);

        imMatches = [imMatches [frameDesc1; frameDesc2]];
        
        matchedFrame = matchedFrame + 1;
    end
    
    figure;
    imshow(imMatches,[]);
  end
end
 
% --------------------------------------------------------------------
% Comprovació del funcionament del calcul descriptor, donats frames
% --------------------------------------------------------------------

 [framesAffOr descAffOr] = featExtractors{2}.extractFeatures(dataset.getImagePath(1));

 n = 10;  % Acotem el número de frames a una mida que permeti entendre què passa
 
 framesAffOr = framesAffOr(:,1:n);
 descAffOr = descAffOr(:,1:n);
 
 % Conversió dels frames orientats, a ellipses no orientades
 ellipses = localFeatures.helpers.frameToEllipse(framesAffOr) ;
 
 % Ara ellipses és de dimensió 5xN
 % Recàlcul de frames i descriptors, passant les ellipses com a paràmetre
 
 [framesAffOr2 descAffOr2] = featExtractors{2}.extractDescriptors(dataset.getImagePath(1),ellipses);

 figure
 imshow(imread(dataset.getImagePath(1)));
 hold on
 vl_plotframe(framesAffOr, 'Color',[1 0 0],'LineWidth',5);
 vl_plotframe(framesAffOr2(:,1:size(framesAffOr,2)), 'Color',[0 1 0],'LineWidth',3,'LineStyle','--');
 vl_plotframe(framesAffOr2(:,(size(framesAffOr,2)+1):end), 'Color',[0 0 1],'LineWidth',1,'LineStyle',':');
 
 size(framesAffOr2)
 

    
    
  
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
  axis([xstart xend 0 maxScore]);
end
end