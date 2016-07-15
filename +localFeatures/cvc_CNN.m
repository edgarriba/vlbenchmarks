classdef cvc_CNN < localFeatures.GenericLocalFeatureExtractor
% localFeatures.TemplateWrapper Feature detector wrapper template
%   Skeleton of image feature detector wrapper.
%
% See also: localFeatures.ExampleLocalFeatureExtractor

% Author: Your Name
   properties (SetAccess=private, GetAccess=public)
    DescrBinPath
    params = cell(10, 1); % arguments buffer

    % Detector options
    Opts = struct(...
      'Name', 'empty');
   end
  
   properties (Constant)
    LuaBinDir = '/home/eriba/software/bmva/cnn_features2d/lua';
    PythonBinDir = '/home/eriba/software/bmva/cnn_features2d/python';
   end
  
  methods
    function obj = cvc_CNN(varargin)

      % Set to true when extractDescriptors implemented
      obj.ExtractsDescriptors = true;

      % Parse class options
      [obj.Opts varargin] = vl_argparse(obj.Opts,varargin);
      
      obj.Name = obj.Opts.Name;

      % Other constructor stuff...

      switch(obj.Name)
        case {'DEEPCOMPARE_siam_liberty'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/siam/siam_liberty.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '256';  % descriptor size
          obj.params{4} = '64';   % patch size
          obj.params{5} = '128'; % batch size
        case {'DEEPCOMPARE_siam2stream_liberty'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/siam2stream/siam2stream_liberty.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '512';  % descriptor size
          obj.params{4} = '64';   % patch size
          obj.params{5} = '1280'; % batch size
        case {'DEEPCOMPARE_2ch2stream_liberty'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/2ch2stream/2ch2stream_liberty.t7';
          obj.params{2} = '2';    % num. input channels
          obj.params{3} = '768';  % descriptor size
          obj.params{4} = '64';   % patch size
          obj.params{5} = '1280'; % batch size
        case {'WLRN_NET_VASSILEIOS'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/wlrn/liberty_wlrn.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '64';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        %% Tfeat MARGIN
        case {'TFeat_MARGIN_E01'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch1.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_E05'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch5.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_E10'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch10.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_E20'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch20.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_E30'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch30.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_E40'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch40.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_E50'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch50.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_E60'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin/liberty_NONORM_net_128_epoch60.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        %% Tfeat MARGIN_STAR
        case {'TFeat_MARGIN_STAR_E01'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch1.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_STAR_E05'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch5.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_STAR_E10'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch10.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_STAR_E20'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch20.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_STAR_E30'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch30.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_STAR_E40'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch40.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_STAR_E50'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch50.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_MARGIN_STAR_E60'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/margin_star/liberty_NONORM_net_128_epoch60.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        %% Tfeat SOFTMAX
        case {'TFeat_SOFTMAX_E01'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch1.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_E05'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch5.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_E10'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch10.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_E20'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch20.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_E30'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch30.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_E40'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch40.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_E50'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch50.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_E60'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax/liberty_NONORM_net_128_epoch60.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        %% Tfeat SOFTMAX_STAR
        case {'TFeat_SOFTMAX_STAR_E01'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch1.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_STAR_E05'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch5.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_STAR_E10'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch10.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_STAR_E20'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch20.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_STAR_E30'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch30.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_STAR_E40'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch40.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_STAR_E50'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch50.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'TFeat_SOFTMAX_STAR_E60'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/tfeat/softmax_star/liberty_NONORM_net_128_epoch60.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'IRI_NET'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'iri_net.sh')];
          obj.params{1} = 'iri_net.lua';
          obj.params{2} = 'torch/iri_net';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '64';   % patch size
          obj.params{6} = '1280'; % batch size
        case {'WLRN_NET'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'pnnet.sh')];
          obj.params{1} = 'wlrn.lua';
          obj.params{2} = 'torch/wlrn';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '64';   % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '128';  % batch size
        case {'PNNet_liberty'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'pnnet.sh')];
          obj.params{1} = 'pnnet.lua';
          obj.params{2} = 'torch/pnnet';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '1280'; % batch size
        case {'PNNet_liberty_NORM'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'pnnet.sh')];
          obj.params{1} = 'pnnet.lua';
          obj.params{2} = 'torch/pnnet_norm';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '1280'; % batch size
        case {'PNNet_liberty_NONORM'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'pnnet.sh')];
          obj.params{1} = 'pnnet.lua';
          obj.params{2} = 'torch/pnnet_nonorm';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '1280'; % batch size
        case {'PNNet_liberty_L2'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'pnnet.sh')];
          obj.params{1} = 'pnnet.lua';
          obj.params{2} = 'torch/pnnet';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '1280'; % batch size
        case {'PNNet_theano_liberty'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.PythonBinDir, 'matchnet.sh')];
          obj.params{1} = 'pnnet_lasagne.py';
          obj.params{2} = 'theano/pnnet';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '128'; % batch size
        case {'PNNET_RATIO'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.LuaBinDir, 'deepcompare.sh')];
          obj.params{1} = 'torch/pnnet/pnnet-liberty-ratio.t7';
          obj.params{2} = '1';    % num. input channels
          obj.params{3} = '128';  % descriptor size
          obj.params{4} = '32';   % patch size
          obj.params{5} = '128'; % batch size
        case {'PNNet_margin_triplets'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.PythonBinDir, 'matchnet.sh')];
          obj.params{1} = 'pnnet_margin_triplets.py';
          obj.params{2} = 'theano/pnnet';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '128'; % batch size
        case {'PNNet_theano_liberty_l2'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.PythonBinDir, 'matchnet.sh')];
          obj.params{1} = 'pnnet_lasagne.py';
          obj.params{2} = 'theano/pnnet';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '128';  % descriptor size
          obj.params{5} = '32';   % patch size
          obj.params{6} = '128'; % batch size          
        case {'MATCHNet_liberty'}
          obj.DescrBinPath = ['bash ' fullfile(localFeatures.cvc_CNN.PythonBinDir, 'matchnet.sh')];
          obj.params{1} = 'matchnet.py';
          obj.params{2} = 'caffe/matchnet';
          obj.params{3} = '1';    % num. input channels
          obj.params{4} = '4096';  % descriptor size
          obj.params{5} = '64';   % patch size
          obj.params{6} = '1280'; % batch size         
       case {'CAFFE_alexnet'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'caffe_simple');
          obj.params{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/alexnet/deploy.prototxt');
          obj.params{2} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/alexnet/bvlc_alexnet.caffemodel');
          obj.params{3} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/alexnet/imagenet_mean.binaryproto');
          obj.params{4} = 'fc8';  % blob feature
          obj.params{5} = '256';  % batch size
        case {'CAFFE_VGG16'}
          obj.DescrBinPath = fullfile(localFeatures.cvc_CNN.BinDir, 'caffe_simple');
          obj.params{1} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt');
          obj.params{2} = fullfile(localFeatures.cvc_CNN.NetworksDir, 'caffe/vgg16/VGG_ILSVRC_16_layers.caffemodel');
          obj.params{3} = 'NULL'; % mean file
          obj.params{4} = 'conv5_3';  % blob feature
          obj.params{5} = '10';   % batch size
        otherwise
          error('ERROR: %s, not supported by cvc_CNN', varargin{1});
      end
    end

    function [frames descriptors] = extractDescriptors(obj, imagePath, frames, featureType)
      import localFeatures.helpers.*;

      % Code to extract descriptors of given frames
      if isempty(frames), descriptors = []; return; end;

      [framesFile, tmpDir, name, ext] = localFeatures.helpers.getFramesFile(...
        imagePath, featureType);

      % call binary and extract descriptors
       descriptors = computeDescriptors(obj, tmpDir, framesFile);
      
      obj.storeFeatures(imagePath, frames, descriptors);
    end

    function descriptors = computeDescriptors(obj, tmpDir, framesFile)
      import localFeatures.*;
      outDescFile = fullfile(tmpDir, strcat(obj.Name ,'_descs.txt'));
      % Prepare the options
      descrArgs = cell(sum(~cellfun(@isempty, obj.params)) + 2, 1);
      descrArgs{1} = framesFile;
      descrArgs{2} = outDescFile;
      for i=1:sum(~cellfun(@isempty, obj.params))
            descrArgs{2+i} = obj.params{i};
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
      obj.debug('Loading Descriptors from text file...');
      descriptors = load(outDescFile)';
      if strcmp(obj.Name, 'PNNet_liberty_L2') || ...
         strcmp(obj.Name, 'PNNet_theano_liberty_l2')
          descriptors = normc(descriptors);
      end
      
      % normalize descriptors by columns
      descriptors = normc(descriptors);
      descriptors = sqrt(descriptors);

      obj.debug('Descriptors computed in %gs', elapsedTime);
    end

    function signature = getSignature(obj)
      % Code for generation of detector unique signature
      signature = [obj.Name,';',...
        helpers.fileSignature(mfilename('fullpath'))];
    end
  end
end
