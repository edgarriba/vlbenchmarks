classdef cvc_CNN < localFeatures.GenericLocalFeatureExtractor
% localFeatures.TemplateWrapper Feature detector wrapper template
%   Skeleton of image feature detector wrapper.
%
% See also: localFeatures.ExampleLocalFeatureExtractor

% Author: Your Name
   properties (SetAccess=private, GetAccess=public)
    DescrBinPath
    NetworkFile = cell(4, 1);
    % Detector options
    Opts = struct(...
      'option1', 'option1_value'... % A detector option
      );
  end
  properties (Constant, Hidden)
    BinDir = '/home/eriba/software/bmva/build/experiment1';
    NetworksDir =  '/home/eriba/software/bmva/experiment1/data';
  end

  methods
    function obj = cvc_CNN(varargin)
     
      % Set to true when extractDescriptors implemented
      obj.ExtractsDescriptors = true;
      obj.Name = strcat('CVC_', varargin{1}); % Name of the wrapper
      %obj.Type = varargin{1};

      % Parse class options
      %obj.Opts = vl_argparse(obj.Opts,varargin)

      % Other constructor stuff...
      
      switch(varargin{1})
        case {'TORCH_siam2stream_desc_notredame'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'torch_simple');
          obj.NetworkFile{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'torch/siam2stream_desc_notredame.bin');
        case {'TORCH_siam_desc_notredame'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'torch_simple');
          obj.NetworkFile{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'torch/siam_desc_notredame.bin');
        case {'LUA_iri'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.NetworksDir, 'lua/lua_net.sh')];
        case {'CAFFE_alexnet'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'caffe_simple');
          obj.NetworkFile{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/alexnet/deploy.prototxt');
          obj.NetworkFile{2} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/alexnet/bvlc_alexnet.caffemodel');
          obj.NetworkFile{3} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/alexnet/imagenet_mean.binaryproto');
          obj.NetworkFile{4} = 'fc7';
        case {'CAFFE_2ch'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'caffe_simple');
          obj.NetworkFile{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/cvpr2015networks-caffe/2ch/liberty_deploy.txt');
          obj.NetworkFile{2} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/cvpr2015networks-caffe/2ch/liberty.caffemodel');
          obj.NetworkFile{3} = 'NULL'; % mean file
          obj.NetworkFile{4} = 'caffe.InnerProduct_9'; % blob feature name
        case {'CAFFE_2chdeep'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'caffe_simple');
          obj.NetworkFile{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/cvpr2015networks-caffe/2chdeep/liberty_deploy.txt');
          obj.NetworkFile{2} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/cvpr2015networks-caffe/2chdeep/liberty.caffemodel');
          obj.NetworkFile{3} = 'NULL';  % mean file
           obj.NetworkFile{4} = 'caffe.Flatten_15'; % blob feature name
        case {'CAFFE_siam'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'caffe_simple');
          obj.NetworkFile{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/cvpr2015networks-caffe/siam/liberty_deploy.txt');
          obj.NetworkFile{2} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/cvpr2015networks-caffe/siam/liberty.caffemodel');
          obj.NetworkFile{3} = 'NULL';  % mean file
           obj.NetworkFile{4} = 'caffe.InnerProduct_22'; % blob feature name
        otherwise
          error('ERROR: %s, not supported by cvc_CNN', varargin{1});
      end
    end

    function [frames descriptors] = extractDescriptors(obj, imagePath, frames)
      % Code to extract descriptors of given frames
      if isempty(frames), descriptors = []; return; end;
    
      % compute normalized patches
      [tmpDir framesFile] = computeNormalizedPatcheS(obj, imagePath, frames);
      % call binary and extract descriptors
      descriptors = computeDescriptors(obj, tmpDir, framesFile);
      % clean temporal files
      rmdir(tmpDir,'s');

    end
    
    function descriptors = computeDescriptors(obj, tmpDir, framesFile)
      import localFeatures.*;
      outDescFile = fullfile(tmpDir, strcat(obj.Name ,'_descs.txt'));
      % Prepare the options
      descrArgs = cell(sum(~cellfun(@isempty, obj.NetworkFile)) + 2, 1);
      descrArgs{1} = framesFile;
      descrArgs{2} = outDescFile;
      for i=1:sum(~cellfun(@isempty, obj.NetworkFile))
            descrArgs{2+i} = obj.NetworkFile{i};
      end
      descrArgs = strjoin(descrArgs);
      descrCmd = [obj.DescrBinPath ' ' descrArgs];
      % call binary
      obj.info('Computing descriptors.');
      startTime = tic;
      obj.debug('Executing: %s', descrCmd);
      [status, msg] = system(descrCmd);
      if status 
        obj.warn('Command %s failed. Trying to rerun.', descrCmd);
      end
      elapsedTime = toc(startTime);
      if status
        error('Computing descriptors failed.\nOffending command: %s\n%s', descrCmd, msg);
      end
      
      % load descriptors
      descriptors = load(outDescFile)';
      obj.debug('Descriptors computed in %gs', elapsedTime);
    end
    
    % Compute normalized patches and  save them in disk.
    % Return the path of the file which includes a list where
    % the patches are saved.
    function [tmpDir framesFile] = computeNormalizedPatcheS(obj, imagePath, frames)
      image = imread(imagePath);
      if(size(image,3)>1), image = rgb2gray(image); end
      image = im2single(image); % If not already in uint8, then convert
      obj.info('Computing descriptors of %d frames.',size(frames,2));
      startTime = tic;
      [frames descriptors] = vl_covdet(image, ...
                                       'Frames', frames, ...
                                       'Descriptor', 'Patch', ...
                                       'PatchResolution', 31, ...
                                       'Verbose') ;
      % setup output path
      [pathstr, name, ext] = fileparts(imagePath);
      tmpDir = fullfile(pwd, pathstr, 'tmp');
      mkdir(tmpDir);
      framesFile = fullfile(tmpDir, strcat(name, '_list.txt'));
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

    function signature = getSignature(obj)
      % Code for generation of detector unique signature
      signature = [obj.NetworkFile{1},';',...
        helpers.fileSignature(mfilename('fullpath'))];
    end
  end
end
