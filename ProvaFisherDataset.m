function [ output_args ] = ProvaFisherDataset( input_args )
%PROVAFISHERDATASET Summary of this function goes here
%   Detailed explanation goes here

close all
clear all

import datasets.*;
import benchmarks.*;
import localFeatures.*;
import benchmarks.helpers.*;
import helpers.*;
      
addpath(genpath(pwd))

% datasetsNames = {'01_graffity','02_autumn_trees', '03_freiburg_center', '04_freiburg_from_munster_crop', '05_freiburg_innenstadt', '09_cool_car', '12_wall', '13_mountains', '14_park_crop', '17_freiburg_munster', '18_graffity', '20_hall2', '21_dog2', '22_small_palace', '23_cat1', '24_cat2'};

datasetsNames = {'17_freiburg_munster', '18_graffity', '20_hall2', '21_dog2', '22_small_palace', '23_cat1', '24_cat2'};

siftDetector = VlFeatSift();

for j=1:length(datasetsNames)

    char(datasetsNames(j))
    dataset = datasets.FisherDataset('Category', char(datasetsNames(j)));
    
    imAPath =  dataset.getImagePath(1);
    [framesA descriptorsA] = siftDetector.extractFeatures(imAPath);
    unorFramesA = localFeatures.helpers.frameToEllipse(framesA) ;
    
    subplot 121;
    imA = imread(imAPath);
    imshow(imA);
    hold on;
%    vl_plotframe(framesA,'g');
    vl_plotframe(unorFramesA,'r');

    for i = 2:dataset.NumImages
        transform = dataset.getTransformation(i);
        imBPath = dataset.getImagePath(i);
        
        %[framesB descriptorsB] = siftDetector.extractFeatures(imBPath);
        
        %unorFramesB = localFeatures.helpers.frameToEllipse(framesB) ;

       % map frames from image A to image B and viceversa
         reprojUnorFramesA = warpEllipse(transform, unorFramesA,...
        'Method','linearise') ;
        
        subplot 122;
        cla;
        imB = imread(imBPath);
        imshow(imB);
        hold on;
        vl_plotframe(reprojUnorFramesA,'r');
%        vl_plotframe(unorFramesA,'r');
        pause(0.1);
               
    end
    
end


end

