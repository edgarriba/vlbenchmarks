classdef SyntheticDataset < datasets.GenericTransfDataset & helpers.Logger...
    & helpers.GenericInstaller
% datasets.SyntheticDataset Wrapper around a synthetic dataset
%   datasets.SyntheticDataset('Option','OptionValue',...) Constructs
%   an object which implements access to a synthetic dataset generated for
%   affine invariant detectors evaluation.
%
%   The dataset is available at: 
%   http://www.cvc.uab.es/~daniel/SyntheticDataset/
%
%   This class perform automatic installation when the dataset data
%   are not available.
%
%   Following options are supported:
%
%   Category :: ['synthGraf']
%     The category within the Synthetic dataset, has to be one of
%     'graf'

% Authors: Daniel Ponsa, adapted from VggAffineDataset.m

% AUTORIGHTS
  properties (SetAccess=private, GetAccess=public)
    Category = 'synthGraf'; % Dataset category
    DataDir; % Image location
    ImgExt; % Image extension
  end

  properties (Constant)
    % All dataset categories
    AllCategories = {'synthGraf'};
  end

  properties (Constant, Hidden)
    % Installation directory
    RootInstallDir = fullfile('data','datasets','SyntheticDataset','');
    % Names of the image transformations in particular categories
    CategoryImageNames = {...
      'Viewpoint angle'... % synthGraf
      };
    % Image labels for particular categories (degree of transf.)
    CategoryImageLabels = {...
      [2 3 4 5 6 7 8]... % synthGraf
      };
    % Root url for dataset tarballs
    RootUrl = 'http://www.cvc.uab.es/~daniel/SyntheticDataset/';
  end

  methods
    function obj = SyntheticDataset(varargin)
      import datasets.*;
      import helpers.*;
      opts.Category = obj.Category;
      [opts varargin] = vl_argparse(opts,varargin);
      [valid loc] = ismember(opts.Category,obj.AllCategories);
      assert(valid,...
        sprintf('Invalid category for synthetic dataset: %s\n',opts.Category));
      obj.DatasetName = ['SyntheticDataset-' opts.Category];
      obj.Category= opts.Category;
      obj.DataDir = fullfile(obj.RootInstallDir,opts.Category,'');
      obj.NumImages = 8;
      obj.checkInstall(varargin);
      ppm_files = dir(fullfile(obj.DataDir,'img*.ppm'));
      pgm_files = dir(fullfile(obj.DataDir,'img*.pgm'));
      if size(ppm_files,1) == 8
        obj.ImgExt = 'ppm';
      elseif size(pgm_files,1) == 8
        obj.ImgExt = 'pgm';
      else
        error('Ivalid dataset image files.');
      end
      obj.ImageNames = obj.CategoryImageLabels{loc};
      obj.ImageNamesLabel = obj.CategoryImageNames{loc};
    end

    function imgPath = getImagePath(obj,imgNo)
      assert(imgNo >= 1 && imgNo <= obj.NumImages,'Out of bounds idx\n');
      imgPath = fullfile(obj.DataDir,sprintf('img%d.%s',imgNo,obj.ImgExt));
    end

    function tfs = getTransformation(obj,imgIdx)
      assert(imgIdx >= 1 && imgIdx <= obj.NumImages,'Out of bounds idx\n');
      if(imgIdx == 1), tfs = eye(3); return; end
      tfs = zeros(3,3);
      [tfs(:,1) tfs(:,2) tfs(:,3)] = ...
         textread(fullfile(obj.DataDir,sprintf('H1to%dp',imgIdx)),...
         '%f %f %f%*[^\n]');
    end
  end

  methods (Access = protected)
    function [urls dstPaths] = getTarballsList(obj)
      import datasets.*;
      installDir = SyntheticDataset.RootInstallDir;
      dstPaths = {fullfile(installDir,obj.Category)};
      urls = {[SyntheticDataset.RootUrl obj.Category '.tar.gz']};
    end
  end
end
