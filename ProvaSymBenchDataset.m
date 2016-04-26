function [ output_args ] = ProvaSymBenchDataset( input_args )
%PROVAFISHERDATASET Summary of this function goes here
%   Detailed explanation goes here

close all
clear all

import datasets.*;
import benchmarks.*;
import localFeatures.*;
import benchmarks.helpers.*;
import Phelpers.*;
      
addpath(genpath(pwd))

datasetsNames = {'arch','arch_easy','bank','bdom','cars',...
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
                 
 datasetsNames = { 'essighaus',  'paintedladies16'};               
                 
siftDetector = VlFeatSift();

for j=1:length(datasetsNames)

    char(datasetsNames(j))
    dataset = datasets.SymBenchDataset('Category', char(datasetsNames(j)));
    
    imAPath =  dataset.getImagePath(1);
    [framesA descriptorsA] = siftDetector.extractFeatures(imAPath);
    unorFramesA = localFeatures.helpers.frameToEllipse(framesA) ;
    

%    vl_plotframe(framesA,'g');


    for i = 2:dataset.NumImages
        transform = dataset.getTransformation(i);
        imBPath = dataset.getImagePath(i);
        
        %[framesB descriptorsB] = siftDetector.extractFeatures(imBPath);
        
        %unorFramesB = localFeatures.helpers.frameToEllipse(framesB) ;

       % map frames from image A to image B and viceversa
         reprojUnorFramesA = warpEllipse(transform, unorFramesA,...
        'Method','linearise') ;
        
        clf
        subplot 121;
        imA = imread(imAPath);
        imshow(imA);
        hold on;
        vl_plotframe(unorFramesA,'r');
        
        subplot 122;
        imB = imread(imBPath);
        imshow(imB);
        hold on;
        vl_plotframe(reprojUnorFramesA,'r');
%        vl_plotframe(unorFramesA,'r');
        pause;
               
    end
    
end


end

