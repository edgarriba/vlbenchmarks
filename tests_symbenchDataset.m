function fischerDatasetTests()
% REPEATABILITYDEMO Demonstrates how to run the repatability benchmark
%   REPEATABILITYDEMO() Runs the repeatability demo.
%
%   REPEATABILITYDEMO(RESULTS_PATH) Run the demo and save the results to
%   path RESULTS_PATH.

% Author: Karel Lenc and Andrea Vedaldi

% AUTORIGHTS

close all
clearvars *

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

% Fischer dataset
datasets_names = {'arch','arch_easy','bank','bdom','cars',...
                  'chinesebuilding','eiffel','essighaus','graffiti',...
                  'londonbridge','madrid','metz','miduomo01',...
                  'miduomo02','montreal','neubrandenburg','notredame',...
                  'notredame12','notredame13','notredame14',...
                  'notredame15','notredame16','paintedladies',...
                  'paintedladies12','paintedladies13','paintedladies14',...
                  'paintedladies15','paintedladies16','paintedladies17',...
                  'paintedladies18','pantheon','pantheon02',...
                  'portcullis','postoffice','riga','sanmarco',...
                  'sanmarco2','stargarder','stargarder3','synagogue',...
                  'taj','tavern','townsquare','trevi02','vatican',...
                  'worldbuilding'};
  
detector = 'harris';
%datasets_names = datasets_names(1);

% Prepare detectors, 
siftDetector = cvc_VlFeatSift();
haraff = VggAffine('detector', 'haraff');

% Prepare descriptors
cvc_cnn1 = cvc_CNN('Name','DEEPCOMPARE_siam_liberty');
cvc_cnn2 = cvc_CNN('Name','DEEPCOMPARE_siam2stream_liberty');
% cvc_cnn3 = cvc_CNN('Name','DEEPCOMPARE_2ch2stream_liberty');
cvc_cnn4 = cvc_CNN('Name','IRI_NET');
cvc_cnn5 = cvc_CNN('Name','PNNet_liberty');
cvc_cnn6 = cvc_CNN('Name','MATCHNet_liberty');
cvc_cnn7 = cvc_CNN('Name','PNNet_theano_liberty');
cvc_cnn8 = cvc_CNN('Name','PNNet_liberty_L2');
cvc_cnn9 = cvc_CNN('Name','PNNet_theano_liberty_l2');
cvc_cnn10 = cvc_CNN('Name','PNNet_liberty_NORM');
cvc_cnn11 = cvc_CNN('Name','PNNet_liberty_NONORM');

% detector is used as descriptor from Harris.
haraffWithSift = DescriptorAdapter(haraff, siftDetector);
haraffWithCNN1 = DescriptorAdapter(haraff, cvc_cnn1);
haraffWithCNN2 = DescriptorAdapter(haraff, cvc_cnn2);
% haraffWithCNN3 = DescriptorAdapter(haraff, cvc_cnn3);
haraffWithCNN4 = DescriptorAdapter(haraff, cvc_cnn4);
haraffWithCNN5 = DescriptorAdapter(haraff, cvc_cnn5);
haraffWithCNN6 = DescriptorAdapter(haraff, cvc_cnn6);
haraffWithCNN7 = DescriptorAdapter(haraff, cvc_cnn7);
haraffWithCNN8 = DescriptorAdapter(haraff, cvc_cnn8);
haraffWithCNN9 = DescriptorAdapter(haraff, cvc_cnn9);
haraffWithCNN10 = DescriptorAdapter(haraff, cvc_cnn10);
haraffWithCNN11 = DescriptorAdapter(haraff, cvc_cnn11);

figure;

featExtractors = {haraffWithSift, ...
                  haraffWithCNN1, haraffWithCNN2,haraffWithCNN4,haraffWithCNN5,...
                  haraffWithCNN6,haraffWithCNN7, haraffWithCNN8,...
                  haraffWithCNN9, haraffWithCNN10};

featExtractors = {haraffWithSift, ...
                  haraffWithCNN4,haraffWithCNN5,...
                  haraffWithCNN7, haraffWithCNN10, haraffWithCNN11};

detectorNames  = {'Harris-Affine SIFT',...
                  'Harris-Affine IRI',...
                  'Harris-Affine PN-Net',...
                  'Harris-Affine PN-Net2',...
                  'Harris-Affine PN-NetNORM',...
                  };

%% First we will compute the descriptors

% for j = 1:numel(datasets_names)
% 
%     char(datasets_names(j))
%     dataset = datasets.FisherDataset('Category',  char(datasets_names(j)));
% end

%% Second we will run the tests

repBenchmark = CVCRepeatabilityBenchmark('Mode','ThresholdBased',...
                                         'maxSamples', 1000,...
                                         'overlapError', 0.5); 

auc = [];
auc2 = zeros(numel(datasets_names), numel(featExtractors), 5);

for j = 1:numel(datasets_names)

    char(datasets_names(j))
    dataset = datasets.SymBenchDataset('Category',  char(datasets_names(j)));

    for d = 1:numel(featExtractors)
      for i = 2:dataset.NumImages
      
        [descriptorMatches, geometryMatches, reprojFrames] = ...
            repBenchmark.testFeatureExtractor(featExtractors{d}, ...
                                              dataset.getTransformation(i), ...
                                              dataset.getImagePath(1), ...
                                              dataset.getImagePath(i));
             % AUC
             if numel(descriptorMatches) > 0
                auc(d,i) = descriptorMatches{1,2}.auc;
                auc2(j,d,i-1) = descriptorMatches{1,2}.auc;
             else
                auc2(j,d,i-1) = 0;
             end

      end
    end
end

save('harris_symbench.mat','auc2');

plotScores(detectorNames, dataset, auc*100,'Area Under Curve');
printScores(detectorNames, auc*100, 'Area Under Curve');

% printScores(detectorNames, matchScore*100, 'Match Score');
% printScores(detectorNames, auc*100, 'Area Under Curve');
% printScores(detectorNames, numMatches, 'Number of matches') ;


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