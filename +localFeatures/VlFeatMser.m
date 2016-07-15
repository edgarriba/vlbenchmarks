classdef VlFeatMser < localFeatures.GenericLocalFeatureExtractor & ...
    helpers.GenericInstaller
% localFeatures.VlFeatMser class to wrap around the VLFeat MSER implementation
%   localFeatures.VlFeatMser('Option','OptionValue',...) constructs an object
%   of the wrapper around the detector.
%
%   The options to the constructor are the same as that for vl_mser
%   See help vl_mser to see those options and their default values.
%
%   See also: vl_mser

% Authors: Karel Lenc, Varun Gulshan

% AUTORIGHTS
  properties (SetAccess=private, GetAccess=public)
    % See help vl_mser for setting parameters for vl_mser
    vlMserArguments;
  end

  methods
    % The constructor is used to set the options for vl_mser call
    % See help vl_mser for possible parameters
    % The varargin is passed directly to vl_mser
    function obj = VlFeatMser(varargin)
      obj.Name = 'VLFeat MSER';
      varargin = obj.checkInstall(varargin);
      obj.vlMserArguments = obj.configureLogger(obj.Name,varargin);
    end

    function [frames] = extractFeatures(obj, imagePath)
      import helpers.*;
      import localFeatures.*;
      frames = obj.loadFeatures(imagePath,false);
      if numel(frames) > 0; return; end;
      startTime = tic;
      obj.info('Computing frames of image %s.',getFileName(imagePath));

      img = imread(imagePath);
      if(size(img,3)>1), img = rgb2gray(img); end
      img = im2uint8(img); % If not already in uint8, then convert

      [xx brightOnDarkFrames] = vl_mser(img,obj.vlMserArguments{:});
      [xx darkOnBrightFrames] = vl_mser(255-img,obj.vlMserArguments{:});

      frames = vl_ertr([brightOnDarkFrames darkOnBrightFrames]);
      sel = frames(3,:).*frames(5,:) - frames(4,:).^2 >= 1 ;
      frames = frames(:, sel) ;
      % HACK HACK HACK 
      [frames tmpDir framesFile] = computeNormalizedPatcheS(obj, imagePath, frames);
      % END HACK HACK HACK
      timeElapsed = toc(startTime);
      obj.debug('%d Frames from image %s computed in %gs',...
        size(frames,2),getFileName(imagePath),timeElapsed);
      obj.storeFeatures(imagePath, frames, []);
    end

    function sign = getSignature(obj)
      signList = {helpers.VlFeatInstaller.getBinSignature('vl_mser'),...
                  helpers.cell2str(obj.vlMserArguments)};
      sign = helpers.cell2str(signList);
    end
    
    % Compute normalized patches and  save them in disk.
    % Return the path of the file which includes a list where
    % the patches are saved.
    function [frames tmpDir framesFile] = computeNormalizedPatcheS(obj, imagePath, frames)
      % setup output path
      [pathstr, name, ext] = fileparts(imagePath);
      tmpDir = fullfile(pwd, pathstr, 'tmp');
      framesFile = fullfile(tmpDir, strcat(name, '_list.txt'));
      if (exist(framesFile, 'file') == 2)
        obj.debug('Already computed normalized frames');
        return;
      end 
      if (~(exist(tmpDir, 'dir') == 7)), mkdir(tmpDir); end 
      % extract rectified frames
      image = imread(imagePath);
      if(size(image,3)>1), image = rgb2gray(image); end
      image = im2single(image); % If not already in uint8, then convert
      obj.info('Computing normalized frames %d.',size(frames,2));
      startTime = tic;
      [frames descriptors] = vl_covdet(image, ...
                                       'Frames', frames, ...
                                       'Descriptor', 'Patch', ...
                                       'PatchResolution', 31, ...
                                       'EstimateAffineShape', true, ...
                                       'EstimateOrientation', true, ...
                                       'Verbose');
      fileID = fopen(framesFile, 'w');
      % convert descriptor to squared patches
      for i=1:size(descriptors, 2)
          % we assume that patches are squared
          N = sqrt(size(descriptors(:,i), 1));
          patch  = reshape(descriptors(:,i), N, N);
          fname = strcat(strcat(name, '_', int2str(i)), ext);
          fname = fullfile(tmpDir, fname);
          imwrite(patch, fname);
          fprintf(fileID,'%s\n', fname);
      end
      timeElapsed = toc(startTime);
      obj.debug('Descriptors of %d frames computed in %gs',...
        size(frames, 2), timeElapsed);
    end
  end
  
  methods (Access=protected)
    function deps = getDependencies(obj)
      deps = {helpers.VlFeatInstaller('0.9.14')};
    end
  end
end
