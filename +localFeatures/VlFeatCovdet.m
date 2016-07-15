classdef VlFeatCovdet < localFeatures.GenericLocalFeatureExtractor & ...
    helpers.GenericInstaller
% localFeatures.VlFeatCovdet VLFeat vl_covdet wrapper
%   VlFeatCovdet('OptionName',OptionValue,...) creates new object
%   which wraps around VLFeat covariant image frames detector. All
%   given options defined in the constructor are passed directly to
%   the vl_covdet function when called.
%
%   The options to the constructor are the same as that for vl_covdet
%   See help vl_covdet to see those options and their default values.
%
%   See also: VL_COVDET().

% Authors: Karel Lenc, Varun Gulshan

% AUTORIGHTS

  properties (SetAccess=public, GetAccess=public)
    VlCovdetArguments
  end

  methods
    function obj = VlFeatCovdet(varargin)
      import helpers.*;
      %obj.Name = 'VLFeatCovDet';
      varargin = obj.configureLogger(obj.Name,varargin);
      obj.VlCovdetArguments = obj.checkInstall(varargin);
      obj.Name = strcat('VLFeatCovDet_', char(obj.VlCovdetArguments(2)));
      obj.ExtractsDescriptors = true;
    end

    function [frames descriptors] = extractFeatures(obj, imagePath)
      import helpers.*;
      import localFeatures.helpers.*;

      [frames, descriptors] = obj.loadFeatures(imagePath,nargout > 1);
      if numel(frames) > 0; return; end;
      img = imread(imagePath);
      if (size(img,3)>1), img = rgb2gray(img); end
      img = im2single(img) ;
      startTime = tic;
      if nargout == 1
        obj.info('Computing frames of image %s.',getFileName(imagePath));

        % HACK HACK HACK

%         max_features = 10;
%         new_size = 2048;
%         
%         scale = new_size / max(size(img));
%         img = imresize(img, scale, 'bilinear');
               
        [frames descriptors_, info] = vl_covdet(img, ...
            obj.VlCovdetArguments{:}, 'Verbose');

      image = imread(imagePath);
      if(size(image,3)>1), image = rgb2gray(image); end
      image = im2single(image); % If not already in uint8, then convert
      
      [frames descriptors_] = vl_covdet(image, ...
                                   'Frames', frames, ...
                                   'Descriptor', 'Patch', ...
                                   'PatchResolution', 31, ...
                                   'EstimateAffineShape', true, ...
                                   'EstimateOrientation', true, ...
                                   'Verbose');

        % store extracted patches
        storePatches(obj, imagePath, descriptors_, obj.Name, img, frames);
        
        % END HACK HACK HACK
 
      else
        obj.info('Computing frames and descriptors of image %s.',...
                 getFileName(imagePath));
        [frames descriptors] = vl_covdet(img,...
                                         obj.VlCovdetArguments{:},...
                                         'Verbose', 1);
      end
      timeElapsed = toc(startTime);
            
      obj.debug('%d Frames from image %s computed in %gs',...
                size(frames,2),getFileName(imagePath),timeElapsed);
      obj.storeFeatures(imagePath, frames, descriptors);
    end
    
    function [frames descriptors] = extractDescriptors(obj, imagePath, frames, name)
      image = imread(imagePath);
      if(size(image,3)>1), image = rgb2gray(image); end
      image = im2single(image); % If not already in uint8, then convert
      obj.info('Computing descriptors of %d frames.',size(frames,2));
      startTime = tic;
      
      %% Bug corregit -> si no , sempre retorna SIFT !!!
%       [frames descriptors] = vl_covdet(image, ...
%                                        'Frames', frames, ...
%                                        'Descriptor', 'SIFT', ...
%                                        'Verbose') ;

%       new_size = 2048;
%         
%       scale = new_size / max(size(image));
%       image = imresize(image, scale, 'bilinear');
 
      [frames descriptors] = vl_covdet(image, ...
                                       'Frames', frames, ...
                                       obj.VlCovdetArguments{:},...
                                       'Verbose') ;                                   
      timeElapsed = toc(startTime);
      obj.debug('Descriptors of %d frames computed in %gs',...
        size(frames,2),timeElapsed);
    end

    function sign = getSignature(obj)
      sign = [helpers.VlFeatInstaller.getBinSignature('vl_covdet'),...
              helpers.cell2str(obj.VlCovdetArguments)];
    end
  end

  methods (Access=protected)
    function deps = getDependencies(obj)
      deps = {helpers.VlFeatInstaller('0.9.16')};
    end
  end
end
