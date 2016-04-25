classdef FisherDataset < datasets.GenericTransfDataset & helpers.Logger...
    & helpers.GenericInstaller
% datasets.FisherDataset Wrapper around the fisher datasets
%   which has been first adapted to follow the same organization
%   than the vgg dataset (using Fisher2vlBenchmark.m)
%
%   Following options are supported:
%
%   Category :: ['01_graffity']
%     The category within the VGG dataset, has to be one of
%     '01_graffity','02_autumn_trees', '03_freiburg_center',
%     '04_freiburg_from_munster_crop', '05_freiburg_innenstadt',
%     '09_cool_car', '12_wall', '13_mountains', '14_park_crop',
%     '17_freiburg_munster', '18_graffity', '20_hall2', '21_dog2',
%     '22_small_palace', '23_cat1', '24_cat2'

% Authors: Daniel Ponsa, Edgar Riba

% AUTORIGHTS
  properties (SetAccess=private, GetAccess=public)
    Category = 'graf'; % Dataset category
    DataDir; % Image location
    ImgExt; % Image extension
  end

  properties (Constant)
    % All dataset categories
    AllCategories = {'01_graffity','02_autumn_trees', '03_freiburg_center', '04_freiburg_from_munster_crop', '05_freiburg_innenstadt', '09_cool_car', '12_wall', '13_mountains', '14_park_crop', '17_freiburg_munster', '18_graffity', '20_hall2', '21_dog2', '22_small_palace', '23_cat1', '24_cat2'};
	
  end

  properties (Constant, Hidden)
    % Installation directory
    RootInstallDir = fullfile('data','datasets','FisherDataset','');
    % Names of the image transformations in particular categories
    
    % TBD: Some characteristic of the image content should be stated.
    
    CategoryImageNames = {...
	'01_graffity',...
	'02_autumn_trees',...
	'03_freiburg_center',...
	'04_freiburg_from_munster_crop',...
	'05_freiburg_innenstadt',...
	'09_cool_car',...
	'12_wall',...
	'13_mountains',...
	'14_park_crop',...
	'17_freiburg_munster',...
	'18_graffity',...
	'20_hall2',...
	'21_dog2',...
	'22_small_palace',...
	'23_cat1',...
	'24_cat2'...	
      };
    % Image labels for particular categories (degree of transf.)
    %
    % TBD: the images are just numbered
    CategoryImageLabels = {...
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26],... 
      [2:26]... 
      };
    % Root url for dataset tarballs
    RootUrl = 'http://www.cvc.uab.es/~daniel/FisherDataset/';
  end
    

  methods
    function obj = FisherDataset(varargin)
      import datasets.*;
      import helpers.*;
      opts.Category = obj.Category;
      [opts varargin] = vl_argparse(opts,varargin);
      [valid loc] = ismember(opts.Category,obj.AllCategories);
      assert(valid,...
        sprintf('Invalid category for Fisher dataset: %s\n',opts.Category));
      obj.DatasetName = ['FisherDataset-' opts.Category];
      obj.Category= opts.Category;
      obj.DataDir = fullfile(obj.RootInstallDir,opts.Category,'');
      obj.NumImages = 26;
      obj.checkInstall(varargin);
      ppm_files = dir(fullfile(obj.DataDir,'img*.ppm'));
      pgm_files = dir(fullfile(obj.DataDir,'img*.pgm'));
      if size(ppm_files,1) == 26
        obj.ImgExt = 'ppm';
      elseif size(pgm_files,1) == 26
        obj.ImgExt = 'pgm';
      else
        error('Ivalid dataset image files.');
      end
      obj.ImageNames = obj.CategoryImageLabels{loc};
      obj.ImageNamesLabel = obj.CategoryImageNames{loc};
    end

    function imgPath = getImagePath(obj,imgNo)
      assert(imgNo >= 1 && imgNo <= obj.NumImages,'Out of bounds idx\n');
      imgPath = fullfile(obj.DataDir,sprintf('img%02d.%s',imgNo,obj.ImgExt));
    end

    function tfs = getTransformation(obj,imgIdx)
      assert(imgIdx >= 1 && imgIdx <= obj.NumImages,'Out of bounds idx\n');
      if(imgIdx == 1), tfs = eye(3); return; end
      tfs = zeros(3,3);
      [tfs(:,1) tfs(:,2) tfs(:,3)] = ...
         textread(fullfile(obj.DataDir,sprintf('H1to%02dp',imgIdx)),...
         '%f %f %f%*[^\n]');
    end
  end

  methods (Access = protected)
    function [urls dstPaths] = getTarballsList(obj)
      import datasets.*;
      installDir = FisherDataset.RootInstallDir;
      dstPaths = {fullfile(installDir,obj.Category)};
      urls = {[FisherDataset.RootUrl obj.Category '.tar.gz']};
    end
  end
end
