classdef CVCRepeatabilityBenchmark < benchmarks.GenericBenchmark ...
    & helpers.Logger & helpers.GenericInstaller

% CVCRepeatabilityBenchmark is based on RepeatabilityBenchmar, with
% somo modifications to evaluate the matching based on descriptors,
% building a precision-recall curve, and computint the Area Under
% Curve to quantify the descriptor performance.

% Next comments are recycled in a big part from
% RepeatabilityBenchmark.m ( Karel Lenc, Andrea Vedaldi )

% benchmarks.CVCRepeatabilityBenchmark Image features repeatability
%   benchmarks.CVCRepeatabilityBenchmark('OptionName',optionValue,...) constructs
%   an object to compute the detector repeatability and the descriptor
%   matching scores as given in [1] and [XXX]. (TBD)
%
%   Using this class is a two step process. First, create an instance of the
%   class specifying any parameter needed in the constructor. Then, use
%   obj.testFeatures() to evaluate the scores given a pair of images, the
%   detected features (and optionally their descriptors), and the homography
%   between the two images.
%
%   Use obj.testFeatureExtractor() to evaluate the test for a given detector
%   and pair of images and being able to cache the results of the test.
%
%   DETAILS ON THE REPEATABILITY AND MATCHING SCORES
%
%   The detector repeatability is calculated for two sets of feature frames
%   FRAMESA and FRAMESB detected in a reference image IMAGEA and a second
%   image IMAGEB. The two images are assumed to be related by a known
%   homography H mapping pixels in the domain of IMAGEA to pixels in the
%   domain of IMAGEB (e.g. static camera, no parallax, or moving camera
%   looking at a flat scene). The homography assumes image coordinates with
%   origin in (0,0).
%
%   A perfect co-variant detector would detect the same features in both
%   images regardless of a change in viewpoint (for the features that are
%   visible in both cases). A good detector will also be robust to noise and
%   other distortion. Repeatability is the percentage of detected features
%   that survive a viewpoint change or some other transformation or
%   disturbance in going from IMAGEA to IMAGEB.
%
%   More in detail, repeatability is by default computed as follows:
%
%   1. The elliptical or circular feature frames FRAMEA and FRAMEB,
%      the image sizes SIZEA and SIZEB, and the homography H are
%      obtained.
%
%   2. Features (ellipses or circles) that are fully visible in both
%      images are retained and the others discarded.
%
%   3. For each pair of feature frames A and B, the normalised overlap
%      measure OVERLAP(A,B) is computed. This is defined as the ratio
%      of the area of the intersection over the area of the union of
%      the ellpise/circle FRAMESA(:,A) and FRAMES(:,B) reprojected on
%      IMAGEA by the homography H. 

%      Q -(PER QUÃˆ CAL FER AIXÃ’ SEGÃœENT ?)
%
%      Furthermore, after reprojection the
%      size of the ellipses/circles are rescaled so that FRAMESA(:,A)
%      has an area equal to the one of a circle of radius 30
%      pixels.

%      Q - (PERQUÃˆ FAN GRAF ? PER NO ASSOCIAR UNA MATEIXA ELIPSE A 2
%      ELIPSES ?
%
%   4. Feature are matched optimistically. A candidate match (A,B) is
%      created for every pair of features A,B such that the
%      OVERLAP(A,B) is larger than a certain threshold (defined as 1 -
%      OverlapError) and weighted by OVERLAP(A,B). Then, the final set
%      of matches M={(A,B)} is obtained by performing a greedy
%      bipartite matching between in the weighted graph
%      thus obtained. Greedy means that edges are assigned in order
%      of decreasing overlap.
%
%   5. Repeatability is defined as the ratio of the number of matches
%      M thus obtained and the minimum of the number of features in
%      FRAMESA and FRAMESB:
%
%                                    |M|
%        repeatability = -------------------------.
%                        min(|framesA|, |framesB|)
%
%   RepeatabilityBenchmark can compute the descriptor matching score
%   too (see the 'Mode' option). To define this, a second set of 
%   matches M_d is obtained similarly to the previous method, except 
%   that the descriptors distances are used in place of the overlap, 
%   no threshold is involved in the generation of candidate matches, and 
%   these are selected by increasing descriptor distance rather than by
%   decreasing overlap during greedy bipartite matching. Then the
%   descriptor matching score is defined as:
%
%                              |inters(M,M_d)|
%        matching-score = -------------------------.
%                         min(|framesA|, |framesB|)
%

%
%   6. An additional criterion to judge descriptor has been
%   added. The nearest neighbor association NNA between descriptors of A
%   and B is established applying the L2 norm (descriptor distance
%   considered). Then, NNAs are sorted from its descriptor
%   distance, and a precision-recall curve is computed, by setting
%   each time the matching threshold equal to the next association
%   to be considered. Once all NNAs are considered good matches,
%   the precision-recall curve is completely computed. Then its AUC
%   is computed.

%   The test behaviour can be adjusted by modifying the following options:
%
%   Mode:: 'Repeatability'
%     Type of score to be calculated. Changes the criteria which are used
%     for finding one-to-one matches between image features.
%
%   DETECTOR BENCHMARK
%
%     'Repeatability'
%       Match frames geometry only. 
%       Corresponds to detector repeatability measure in [1].
%       One frame in A is associated to just another frame in B 1-1.
%       A frame of B already associated, is not associated again.
%       Based on greedy bipartite matching. Every node of each set (A or B)
%       in the bipartite matching is connected to just another node of the
%       other set.
%
%     'RepeatabilityNN'
%       Match frames geometry only. 
%       In this case, each node in A or B can be associtated to multiple
%       nodes in the opposite set, as long as their overlapp fulfills the
%       required level.
%     
%   DESCRIPTOR BENCHMARK
%
%     'ThresholdBased'
%       Associates frames of A and B, if distance between their descriptor
%       is below a threshold.
%       With this criterion, one descriptor can have several matches. 
%       To validate this method, the GT is based on 'RepeatabilityN2N'
%
%     'NearestNeighbor'
%       Frame a_i is matched with b_j if 
%        >> D_ij is the minimum distace between a_i and any other node of B
%        >> Distance is below a threshold
%       Each node in A has only one match.
%       Each node in B should have just only one match ?????
%       To validate this method, the GT is based on 'Repeatability'
%       
%     'NearestNeighborRatio'
%       Frame a_i is matched with b_j if the ratio between the first and
%       second nearest neighbor is below a threshold.
%       Each node in A has only one match.
%       Each node in B should have just only one match ?????
%       To validate this method, the GT is based on 'Repeatability'
% 
%
%     'MatchingScore'
%       Match frames geometry and frame descriptors.
%       Corresponds to detector matching score in [1], but 
%       computing the performance using  the precision - recall
%       curve of NNA, and then summarises the result computing the
%       AUV.
%
%     'DescMatchingScore'
%        Match frames only based on their descriptors.
%
%
%   OverlapError:: 0.4
%     Maximal overlap error of frames to be considered as
%     correspondences.
%
%   NormaliseFrames:: true
%     Normalise the frames to constant scale (defaults is true for
%     detector repeatability tests, see Mikolajczyk et. al 2005).
%
%   NormalisedScale:: 30
%     When frames scale normalisation applied, fixed scale to which it is
%     normalised to.
%
%   CropFrames:: true
%     Crop the frames out of overlapping regions (regions present in both
%     images).
%
%   Magnification:: 3
%     When frames are not normalised, this parameter is magnification
%     applied to the input frames. Usually is equal to magnification
%     factor used for descriptor calculation.
%
%   WarpMethod:: 'linearise'
%     Numerical method used for warping ellipses. Available mathods are
%     'standard' and 'linearise' for precise reproduction of IJCV2005 
%     benchmark results.
%
%   DescriptorsDistanceMetric:: 'L2'
%     Distance metric used for matching the descriptors. See
%     documentation of vl_alldist2 for details.
%
%   See also: datasets.VggAffineDataset, vl_alldist2
%
%   REFERENCES
%   [1] K. Mikolajczyk, T. Tuytelaars, C. Schmid, A. Zisserman,
%       J. Matas, F. Schaffalitzky, T. Kadir, and L. Van Gool. A
%       comparison of affine region detectors. IJCV, 1(65):43â€“72, 2005.

% Authors: Karel Lenc, Andrea Vedaldi

% AUTORIGHTS

  properties
    Opts = struct(...
      'overlapError', 0.4,...
      'normaliseFrames', true,...
      'cropFrames', true,...
      'magnification', 3,...
      'warpMethod', 'linearise',...
      'mode', 'repeatability',...
      'descriptorsDistanceMetric', 'L2',...
      'normalisedScale', 30);
  end

  properties(Constant, Hidden)
    KeyPrefix = 'repeatability';
    %
        
    Modes = {'repeatability','repeatabilityn2n','thresholdbased',...
             'nearestneighbor','nearestneighborratio','matchingscore',...
             'descmatchingscore'};
    
    ModesOpts = containers.Map(benchmarks.CVCRepeatabilityBenchmark.Modes,...
      {struct('matchGeometry',true,'matchDescs',false),...
       struct('matchGeometry',true,'matchDescs',false),...
       struct('matchGeometry',true,'matchDescs',true),...
       struct('matchGeometry',true,'matchDescs',true),...
       struct('matchGeometry',true,'matchDescs',true),...
       struct('matchGeometry',true,'matchDescs',true),...
       struct('matchGeometry',false,'matchDescs',true)});
  end

  methods
    function obj = CVCRepeatabilityBenchmark(varargin)
      import benchmarks.*;
      import helpers.*;
      obj.BenchmarkName = 'repeatability';
      if numel(varargin) > 0
        [obj.Opts varargin] = vl_argparse(obj.Opts,varargin);
        obj.Opts.mode = lower(obj.Opts.mode);
        if ~ismember(obj.Opts.mode, obj.Modes)
          error('Invalid mode %s.',obj.Opts.mode);
        end
      end
      varargin = obj.configureLogger(obj.BenchmarkName,varargin);
      obj.checkInstall(varargin);
    end

    function [descriptorMatches, geometryMatches, reprojFrames] = ...
        testFeatureExtractor(obj, featExtractor, tf, imageAPath, imageBPath)
    
      % ACTUALITZAR INFORMACIÓ HELP
    
      % testFeatureExtractor Image feature extractor repeatability
      %   REPEATABILITY = obj.testFeatureExtractor(FEAT_EXTRACTOR, TF,
      %   IMAGEAPATH, IMAGEBPATH) computes the repeatability REP of a image
      %   feature extractor FEAT_EXTRACTOR and its frames extracted from
      %   images defined by their path IMAGEAPATH and IMAGEBPATH whose
      %   geometry is related by the homography transformation TF.
      %   FEAT_EXTRACTOR must be a subclass of
      %   localFeatures.GenericLocalFeatureExtractor.
      %
      %   [REPEATABILITY, NUMMATCHES] = obj.testFeatureExtractor(...) 
      %   returns also the total number of feature matches found.
      %
      %   [REP, NUMMATCHES, REPR_FRAMES, MATCHES] =
      %   obj.testFeatureExtractor(...) returns cell array REPR_FRAMES which
      %   contains reprojected and eventually cropped frames in
      %   format:
      %
      %   REPR_FRAMES = {CFRAMES_A,CFRAMES_B,REP_CFRAMES_A,REP_CFRAMES_B}
      %
      %   where CFRAMES_A are (cropped) frames detected in the IMAGEAPATH
      %   image REP_CFRAMES_A are CFRAMES_A reprojected to the IMAGEBPATH
      %   image using homography TF. Same hold for frames from the secons
      %   image CFRAMES_B and REP_CFRAMES_B.
      %   MATCHES is an array of size [size(CFRAMES_A),1]. Two frames are
      %   CFRAMES_A(k) and CFRAMES_B(l) are matched when MATCHES(k) = l.
      %   When frame CFRAMES_A(k) is not matched, MATCHES(k) = 0.
      %
      %   This method caches its results, so that calling it again will not
      %   recompute the repeatability score unless the cache is manually
      %   cleared.
      %
      %   See also: benchmarks.RepeatabilityBenchmark().
      import benchmarks.*;
      import helpers.*;

      obj.info('Comparing frames from det. %s and images %s and %s.',...
          featExtractor.Name,getFileName(imageAPath),...
          getFileName(imageBPath));

      imageASign = helpers.fileSignature(imageAPath);
      imageBSign = helpers.fileSignature(imageBPath);
      imageASize = helpers.imageSize(imageAPath);
      imageBSize = helpers.imageSize(imageBPath);
      resultsKey = cell2str({obj.KeyPrefix, obj.getSignature(), ...
        featExtractor.getSignature(), imageASign, imageBSign});
    
      cachedResults = obj.loadResults(resultsKey);

      % When detector does not cache results, do not use the cached data
      if isempty(cachedResults) || ~featExtractor.UseCache
        if obj.ModesOpts(obj.Opts.mode).matchDescs
%          [framesA descriptorsA] = featExtractor.extractFeatures(imageAPath);
%          [framesB descriptorsB] = featExtractor.extractFeatures(imageBPath);
          
           [framesA descriptorsA framesB descriptorsB] = obj.DetectAndDescribeFrames(...
               featExtractor,imageAPath,imageBPath);
 
           [descriptorMatches, geometryMatches, reprojFrames] = obj.testFeatures(...
                tf, imageASize, imageBSize, framesA, framesB,...
                descriptorsA, descriptorsB);
	   
        else
%          [framesA] = featExtractor.extractFeatures(imageAPath);
%          [framesB] = featExtractor.extractFeatures(imageBPath);
          
          [framesA framesB] = obj.DetectFrames(...
               featExtractor,imageAPath,imageBPath);
          
          [descriptorMatches, geometryMatches, reprojFrames] = ...
            obj.testFeatures(tf,imageASize, imageBSize,framesA, framesB);
        end
        if featExtractor.UseCache
          results = {descriptorMatches, geometryMatches, reprojFrames};
          obj.storeResults(results, resultsKey);
        end
      else
        [descriptorMatches, geometryMatches, reprojFrames] = cachedResults{:};
        obj.debug('Results loaded from cache');
      end

    end

    function [framesA descriptorsA framesB descriptorsB] = ...
            DetectAndDescribeFrames(obj,featExtractor,imageAPath,imageBPath)
                    
        [framesA descriptorsA] = featExtractor.extractFeatures(imageAPath);
        [framesB descriptorsB] = featExtractor.extractFeatures(imageBPath);   
    end
 
    function [framesA framesB ] = ...
            DetectFrames(obj,featExtractor,imageAPath,imageBPath)
        
        [framesA] = featExtractor.extractFeatures(imageAPath);
        [framesB] = featExtractor.extractFeatures(imageBPath);
    end
    
    function [descriptorMatches, geometryMatches, reprojFrames] = ...
        testFeatures(obj, tf, imageASize, imageBSize, framesA, framesB, ...
        descriptorsA, descriptorsB)
      
      % HELP A ACTUALITZAR.
      % descriptorMatches. Array amb estructura recopilant resultats per
      % una estrategia de posar en correspondència regions
      % NN-121, NN-N21, NNR-121, NNR-N21, thrBased-N2N
      % (Aclarir camps de l'estructura)
    
      % TEXT OBSOLET
      % testFeatures Compute repeatability of given image features
      %   [SCORE AUC NUM_MATCHES] = obj.testFeatures(TF, IMAGE_A_SIZE,
      %   IMAGE_B_SIZE, FRAMES_A, FRAMES_B, DESCS_A, DESCS_B) Compute
      %   matching score SCORE between frames FRAMES_A and FRAMES_B
      %   and their descriptors DESCS_A and DESCS_B which were
      %   extracted from pair of images with sizes IMAGE_A_SIZE and
      %   IMAGE_B_SIZE which geometry is related by homography TF.
      %   AUC is the area under curve of the precision-call curve,
      %   NUM_MATCHES is number of matches which is calculated
      %   according to object settings.
      %
      %   [SCORE, AUC, NUM_MATCHES, MATCHES, REPR_FRAMES] =
      %   obj.testFeatures(...) returns cell array REPR_FRAMES which
      %   contains reprojected and eventually cropped frames in
      %   format:
      %
      %   REPR_FRAMES = {CFRAMES_A,CFRAMES_B,REP_CFRAMES_A,REP_CFRAMES_B}
      %
      %   where CFRAMES_A are (cropped) frames detected in the IMAGEAPATH
      %   image REP_CFRAMES_A are CFRAMES_A reprojected to the IMAGEBPATH
      %   image using homography TF. Same hold for frames from the secons
      %   image CFRAMES_B and REP_CFRAMES_B.
      %   MATCHES is an array of size [size(CFRAMES_A),1]. Two frames are
      %   CFRAMES_A(k) and CFRAMES_B(l) are matched when MATCHES(k) = l.
      %   When frame CFRAMES_A(k) is not matched, MATCHES(k) = 0.
      
                 
      import benchmarks.helpers.*;
      import helpers.*;

      obj.info('Computing score between %d/%d frames.',...
          size(framesA,2),size(framesB,2));
      matchGeometry = obj.ModesOpts(obj.Opts.mode).matchGeometry;
      matchDescriptors = obj.ModesOpts(obj.Opts.mode).matchDescs;

      if isempty(framesA) || isempty(framesB)
        geometryMatches121 = zeros(1,size(framesA,2)); reprojFrames = {};
        geometryMatchesN2N = zeros(size(framesA,2),size(framesB,2));
        obj.info('Nothing to compute.');
        return;
      end
      if exist('descriptorsA','var') && exist('descriptorsB','var')
        if size(framesA,2) ~= size(descriptorsA,2) ...
            || size(framesB,2) ~= size(descriptorsB,2)
          obj.error('Number of frames and descriptors must be the same.');
        end
      elseif matchDescriptors
        obj.error('Unable to match descriptors without descriptors.');
      end

      startTime = tic;
      normFrames = obj.Opts.normaliseFrames;
      overlapError = obj.Opts.overlapError;
      overlapThresh = 1 - overlapError;

      % To compute the overlap between frames, first frames are
      % converted from any supported format to unortiented
      % ellipses (for uniformity)
      unorFramesA = localFeatures.helpers.frameToEllipse(framesA) ;
      unorFramesB = localFeatures.helpers.frameToEllipse(framesB) ;

      % map frames from image A to image B and viceversa
      reprojUnorFramesA = warpEllipse(tf, unorFramesA,...
        'Method',obj.Opts.warpMethod) ;
      reprojUnorFramesB = warpEllipse(inv(tf), unorFramesB,...
        'Method',obj.Opts.warpMethod) ;

      % optionally remove frames that are not fully contained in
      % both images
      if obj.Opts.cropFrames
        % find frames fully visible in both images
        bboxA = [1 1 imageASize(2)+1 imageASize(1)+1] ;
        bboxB = [1 1 imageBSize(2)+1 imageBSize(1)+1] ;

        visibleFramesA = isEllipseInBBox(bboxA, unorFramesA ) & ...
          isEllipseInBBox(bboxB, reprojUnorFramesA);

        visibleFramesB = isEllipseInBBox(bboxA, reprojUnorFramesB) & ...
          isEllipseInBBox(bboxB, unorFramesB );

        % Crop frames outside overlap region
        framesA = framesA(:,visibleFramesA);
        unorFramesA = unorFramesA(:,visibleFramesA);
        reprojUnorFramesA = reprojUnorFramesA(:,visibleFramesA);
	
        framesB = framesB(:,visibleFramesB);
        unorFramesB = unorFramesB(:,visibleFramesB);
        reprojUnorFramesB = reprojUnorFramesB(:,visibleFramesB);
	
        if isempty(framesA) || isempty(framesB)
          matches = zeros(size(framesA,2)); reprojFrames = {};
          matchesN2N = matches;
          return;
        end

        if matchDescriptors
          descriptorsA = descriptorsA(:,visibleFramesA);
          descriptorsB = descriptorsB(:,visibleFramesB);
        end
      end

      if ~normFrames
        % When frames are not normalised, account the descriptor region
        magFactor = obj.Opts.magnification^2;
        unorFramesA = [unorFramesA(1:2,:); unorFramesA(3:5,:).*magFactor];
        reprojUnorFramesB = [reprojUnorFramesB(1:2,:); ...
          reprojUnorFramesB(3:5,:).*magFactor];
      end

      reprojFrames = {unorFramesA,unorFramesB,reprojUnorFramesA, ...
		      reprojUnorFramesB};
      
      numFramesA = size(unorFramesA,2);
      numFramesB = size(reprojUnorFramesB,2);

      % Find all ellipse overlaps (in one-to-n array)
      frameOverlaps = fastEllipseOverlap(reprojUnorFramesB, unorFramesA, ...
        'NormaliseFrames',normFrames,'MinAreaRatio',overlapThresh,...
        'NormalisedScale',obj.Opts.normalisedScale);


      if matchGeometry
        % Create an edge between each feature in A and in B
        % weighted by the overlap. Each edge is a candidate match.
        corresp = cell(1,numFramesA);
        for j=1:numFramesA
          numNeighs = length(frameOverlaps.scores{j});
          if numNeighs > 0
            corresp{j} = [j *ones(1,numNeighs); ...
                          frameOverlaps.neighs{j}; ...
                          frameOverlaps.scores{j}];
          end
        end
        corresp = cat(2,corresp{:}) ;
        if isempty(corresp)
          geometryMatches121 = zeros(1,numFramesA); 
          geometryMatchesN2N = zeros(numFramesA,numFramesB);
          return;
        end

        % Remove edges (candidate matches) that have insufficient overlap
        corresp = corresp(:,corresp(3,:) > overlapThresh) ;
        if isempty(corresp)
          geometryMatches121 = zeros(1,numFramesA); 
          geometryMatchesN2N = zeros(numFramesA,numFramesB);
          return;
        end

        % Sort the edgest by decrasing score
        [drop, perm] = sort(corresp(3,:), 'descend');
        corresp = corresp(:, perm);

        % 1-1 matches
        % Approximate the best bipartite matching
        obj.info('Matching frames geometry.');
        geometryMatches121 = greedyBipartiteMatching(numFramesA,...
          numFramesB, corresp(1:2,:)');
               
                   
        % N-N matches
        geometryMatchesN2N = zeros(numFramesA,numFramesB);
        for j=1:size(corresp,2)
            geometryMatchesN2N(corresp(1,j),corresp(2,j)) = corresp(3,j);
        end
      end

      geometryMatches = {geometryMatches121 geometryMatchesN2N};
      
      descriptorMatches = {};
      
      if matchDescriptors
        obj.info('Computing cross distances between all descriptors');
        dists = vl_alldist2(single(descriptorsA),single(descriptorsB),...
          obj.Opts.descriptorsDistanceMetric);
        obj.info('Sorting distances')
        
        copyDist = dists;
        [dists, perm] = sort(dists(:),'ascend');

        % Create list of edges in the bipartite graph
        [aIdx bIdx] = ind2sub([numFramesA, numFramesB],perm(1:numel(dists)));
        edges = [aIdx bIdx];

        % Nearest Neighbor Matching 
        % -------------------------------------       
        % 1-1 matching (frame A_i is matched at one frame B_j and
        % viceversa)
        % Find one-to-one best matches
        obj.info('Matching descriptors.');
        descMatchesNN121 = greedyBipartiteMatching(numFramesA, numFramesB, edges);
                
        for j=1:size(descMatchesNN121,2)
            if (descMatchesNN121(j)~=0)
                distDescMatchesNN121(j) = copyDist(j,descMatchesNN121(j));
            else
                distDescMatchesNN121(j) = inf;
            end
        end
        
        % N-1 matching. A frame B_j can be associated to more that one
        % frame in A. (N-to-1)
        [trash,descMatchesNNN21] = min(copyDist');
        
        for j=1:size(descMatchesNNN21,2)
            if (descMatchesNNN21(j)~=0)
                distDescMatchesNNN21(j) = copyDist(j,descMatchesNNN21(j));
            else
                distDescMatchesNNN21(j) = inf;
            end
        end
        
        
        % Nearest Neighbor Ratio Matching 
        % -------------------------------------
        %  dist_1stNN / dist_2ndNN      
        
        % 1-1 matching (frame A_i is matched at one frame B_j and
        % viceversa)
        % Find one-to-one best matches
        
        tic
        
        descMatchesNNR121 = zeros(1,numFramesA);
        distDescMatchesNNR121 = ones(1,numFramesA);
        
        maxDist = max(copyDist(:));
        acumMinNNR = 0;
        distForNNR = copyDist;
        for j=1:numFramesA
        
            % for each frame A_i, we sort the frames B_j w.r.t. distance
    %        [sortedRowDist,perm] = sort(distForNNR,2,'ascend');
            
            % for each frame A_i, we get the 1st and second frames B_j w.r.t. distance
            % This is 10 times faster than using 'sort'
            
            [firstMinDist, firstMinDistIndex] = min(distForNNR,[],2);
            tmpDistForNNR = distForNNR;            
            matIndexes = sub2ind(size(distForNNR),1:size(distForNNR,1), firstMinDistIndex');
            tmpDistForNNR(matIndexes)= maxDist;
            
            secondMinDist = min(tmpDistForNNR,[],2);
          
            % Select best association between one frame in A and one in B,
            % in terms of (1st-NN / 2nd-NN)
            % NNR = sortedRowDist(:,1)./sortedRowDist(:,2);
            
            NNR = firstMinDist./secondMinDist;
           
            [minNNR, indexA] = min(NNR);
            
            indexB = firstMinDistIndex(indexA);
            
            % When all frames in B has been assigned, all NNR values are 1
            if minNNR < 1
                descMatchesNNR121(1,indexA) = indexB;
                
                            
                % The greedy approach of stablishing matching, eliminating at eah
                % iteration one frame of B as matchable, provoques that the value
                % of the NNR used at each iteration is not indicative of the order
                % of matching established (the first match can have a NNR worse
                % than a posterior match). This may distort the precision-recall
                % curve. To maintain the order of stablishing matchings, in 
                % distDeschMathesNNR121 we register the accumulation of the ratios
                % of all the matchings established up till now.
                
                distDescMatchesNNR121(1,indexA) = acumMinNNR + minNNR;
                
                acumMinNNR = acumMinNNR + minNNR;
            end
            
            % 'Eliminate' the assigned frames in A and B from the distance
            %  matrix. To make things easy, just their distances are set to
            %  a very big number..
                        
            distForNNR(indexA,:) = maxDist;
            distForNNR(:,indexB) = maxDist;
        end

        'temps NNR'
        toc
              
        % N-1 matching. A frame B_j can be associated to more that one
        % frame in A. (N-to-1)
        
        % for each frame A_i, we sort the frames B_j w.r.t. distance
        [sortedRowDist,perm] = sort(copyDist,2,'ascend');
        
        distDescMatchesNNRN21 = (sortedRowDist(:,1)./sortedRowDist(:,2))';        
        descMatchesNNRN21 = perm(:,1)';
        
        
        % Threshold-Based Matching 
        % -------------------------------------
        % A frame in A can be matched with multiples frames in B and
        % viceversa. In fact, to compute the precision-recall curve for all
        % distance thresholds, each frame in A is associated with all
        % frames in B.
        [idFramesB, idFramesA] = meshgrid(1:numFramesB,1:numFramesA);
        distDescMatchesThrBasedN2N = copyDist(:)';
        idFramesB = idFramesB(:);
        idFramesA = idFramesA(:);
        
        [distDescMatchesThrBasedN2N,perm] = sort(distDescMatchesThrBasedN2N,'ascend');
        
        % A list is generated, indicated frames A_i B_j paired.
        descMatchesThrBasedN2N = [idFramesA(perm) , idFramesB(perm)]';
                
        
        % -----------------------------------------------------
        % Recast of the descriptor matching information, to ease the
        % computation of the precision-recall curve and AUC value

        % NN
        % ----
        
        % 1-1
        [pairedMatchesNN121, correctMatchesNN121, sortedDistDescMatchesNN121] = ...
            LabelCorrect121MatchesAndRecast(descMatchesNN121,...
                                         geometryMatches121,...
                                         distDescMatchesNN121);
        
        % N-1
        
        [pairedMatchesNNN21, correctMatchesNNN21, sortedDistDescMatchesNNN21] = ...
            LabelCorrectN21MatchesAndRecast(descMatchesNNN21,...
                                         geometryMatchesN2N,...
                                         distDescMatchesNNN21);
        
        
        % NNR
        % -----
        
        % 1-1
        [pairedMatchesNNR121, correctMatchesNNR121, sortedDistDescMatchesNNR121] = ...
            LabelCorrect121MatchesAndRecast(descMatchesNNR121,...
                                         geometryMatches121,...
                                         distDescMatchesNNR121);      
        % N-1
        
        [pairedMatchesNNRN21, correctMatchesNNRN21, sortedDistDescMatchesNNRN21] = ...
            LabelCorrectN21MatchesAndRecast(descMatchesNNRN21,...
                                         geometryMatchesN2N,...
                                         distDescMatchesNNRN21);
        
        % Thr-based
        % -----------
        
        % N-N
        
        [pairedMatchesThrBasedN2N, correctMatchesThrBasedN2N, sortedDistDescMatchesThrBasedN2N] = ...
            LabelCorrectN2NMatchesAndRecast(descMatchesThrBasedN2N,...
                                         geometryMatchesN2N,...
                                         distDescMatchesThrBasedN2N);
                                     
                                     
        % Precision-recall evaluation
        % -----------------------------
                                     
        [ aucNN121, aucGMNN121, precisionNN121, recallNN121, recallGMNN121] = ...
            PrecisionRecallComputation(correctMatchesNN121, sum(geometryMatches121(:)>0));
                
        [ aucNNN21, aucGMNNN21, precisionNNN21, recallNNN21, recallGMNNN21] = ...
            PrecisionRecallComputation(correctMatchesNNN21, sum(geometryMatchesN2N(:)>0));
        
        [ aucNNR121, aucGMNNR121, precisionNNR121, recallNNR121, recallGMNNR121] = ...
            PrecisionRecallComputation(correctMatchesNNR121, sum(geometryMatches121(:)>0));
        
        [ aucNNRN21, aucGMNNRN21, precisionNNRN21, recallNNRN21, recallGMNNRN21] = ...
            PrecisionRecallComputation(correctMatchesNNRN21, sum(geometryMatchesN2N(:)>0));
        
        [ aucThrBasedN2N, aucGMThrBasedN2N, precisionThrBasedN2N, recallThrBasedN2N, recallGMThrBasedN2N] = ...
            PrecisionRecallComputation(correctMatchesThrBasedN2N, sum(geometryMatchesN2N(:)>0));
              
        % Result organization CODI A OPTIMITZAR !!
        
        res.method = 'NN-121';
        res.correctMatches = correctMatchesNN121;
        res.precision = precisionNN121
        res.recall = recallNN121;
        res.recallGM = recallGMNN121;
        res.auc = aucNN121;
        res.aucGM = aucGMNN121;
        
        obj.info('AUC Method %s : %g', res.method, res.auc);

       
        descriptorMatches{1} = res;
        
        res.method = 'NN-N21';
        res.correctMatches = correctMatchesNNN21;
        res.precision = precisionNNN21
        res.recall = recallNNN21;
        res.recallGM = recallGMNNN21;
        res.auc = aucNNN21;
        res.aucGM = aucGMNNN21;
        
        obj.info('AUC Method %s : %g', res.method, res.auc);
        
        descriptorMatches{2} = res;
        
        res.method = 'NNR-121';
        res.correctMatches = correctMatchesNNR121;
        res.precision = precisionNNR121
        res.recall = recallNNR121;
        res.recallGM = recallGMNNR121;
        res.auc = aucNNR121;
        res.aucGM = aucGMNNR121;
        
        obj.info('AUC Method %s : %g', res.method, res.auc);
        
        descriptorMatches{3} = res;
        
        res.method = 'NNR-N21';
        res.correctMatches = correctMatchesNNRN21;
        res.precision = precisionNNRN21
        res.recall = recallNNRN21;
        res.recallGM = recallGMNNRN21;
        res.auc = aucNNRN21;
        res.aucGM = aucGMNNRN21;
        
        obj.info('AUC Method %s : %g', res.method, res.auc);
        
        descriptorMatches{4} = res;
        
        res.method = 'ThrBased-N2N';
        res.correctMatches = correctMatchesThrBasedN2N;
        res.precision = precisionThrBasedN2N;
        res.recall = recallThrBasedN2N;
        res.recallGM = recallGMThrBasedN2N;
        res.auc = aucThrBasedN2N;
        res.aucGM = aucGMThrBasedN2N;
        
        obj.info('AUC Method %s : %g', res.method, res.auc);
        
        descriptorMatches{5} = res;       
        
if 0        
        figure;
        subplot(2,5,1);
        plot(recallNN121,precisionNN121), xlabel('Recall'), ylabel('Precision');
        title('NN-121');
        subplot(2,5,5+1);
        plot(recallGMNN121,precisionNN121), xlabel('RecallGM'), ylabel('Precision');
        title('NN-121');

        subplot(2,5,2);
        plot(recallNNN21,precisionNNN21), xlabel('Recall'), ylabel('Precision');
        title('NN-N21');
        subplot(2,5,5+2);
        plot(recallGMNNN21,precisionNNN21), xlabel('RecallGM'), ylabel('Precision');
        title('NN-N21');
        
        subplot(2,5,3);
        plot(recallNNR121,precisionNNR121), xlabel('Recall'), ylabel('Precision');
        title('NNR-121');
        subplot(2,5,5+3);
        plot(recallGMNNR121,precisionNNR121), xlabel('RecallGM'), ylabel('Precision');
        title('NNR-121');
        
        subplot(2,5,4);
        plot(recallNNRN21,precisionNNRN21), xlabel('Recall'), ylabel('Precision');
        title('NNR-N21');
        subplot(2,5,5+4);
        plot(recallGMNNRN21,precisionNNRN21), xlabel('RecallGM'), ylabel('Precision');
        title('NNR-N21');            
      
        subplot(2,5,5);
        plot(recallThrBasedN2N,precisionThrBasedN2N), xlabel('Recall'), ylabel('Precision');
        title('ThrBased-N2N');
        subplot(2,5,5+5);
        plot(recallGMThrBasedN2N,precisionThrBasedN2N), xlabel('RecallGM'), ylabel('Precision');
        title('ThrBased-N2N');       
        
        
        figure;
        
        [aucNN121 aucNNN21 aucNNR121 aucNNRN21 aucThrBasedN2N]
        [aucGMNN121 aucGMNNN21 aucGMNNR121 aucGMNNRN21 aucGMThrBasedN2N]
        
        plot([aucNN121 aucNNN21 aucNNR121 aucNNRN21 aucThrBasedN2N],'b-');
        hold on
        plot([aucGMNN121 aucGMNNN21 aucGMNNR121 aucGMNNRN21 aucGMThrBasedN2N],'b--');
        legend('AUC','AUCGM');
        
        figure;
        plot(recallNN121,precisionNN121,'b-');
        xlabel('Recall'), ylabel('Precision');
        hold on;
        plot(recallNNN21,precisionNNN21,'r-');
        plot(recallNNR121,precisionNNR121,'b.');
        plot(recallNNRN21,precisionNNRN21,'r.');
        plot(recallThrBasedN2N,precisionThrBasedN2N,'g-')
        
        legend('NN121','NNN21','NNR121','NNRN21','ThrBased');
        
end;

              
        

       end



    end

    function signature = getSignature(obj)
      signature = helpers.struct2str(obj.Opts);
    end
  end

  methods (Access = protected)
    function deps = getDependencies(obj)
      deps = {helpers.Installer(),helpers.VlFeatInstaller('0.9.14'),...
        benchmarks.helpers.Installer()};
    end
  end

end


% Function that given an association between frames in A and B in the 
% format descMatches(idFrameA) = idFrameB (or 0 if idFrameA has no match),
% recast it to a vector of pairs (idFrameA, idFrameB) containing just the
% effective matches. Each pair is determined if it is a correct match or
% not, if both frames have been also matched geometrically (using the GT
% homography.
% Finally, the computed vectors are sorted with regard to the descriptor
% similariy

% Case 1-1 matches -> we also consider 1-1 geometry matches
% descMatches & geometryMatches are 1xNumFramesA vectors
function [pairedMatches, correctMatches, sortedDistDescMatches] = ...
    LabelCorrect121MatchesAndRecast(descMatches, geometryMatches, distDescMatches)
    
    idFramesA = find(descMatches ~= 0);
    pairedMatches = [idFramesA; descMatches(idFramesA)];
    correctMatches = (descMatches==geometryMatches);
    correctMatches = correctMatches(idFramesA);
    distDescMatches = distDescMatches(idFramesA);
    
    [sortedDistDescMatches,perm] = sort(distDescMatches,'ascend');
    
    pairedMatches = pairedMatches(:, perm);
    correctMatches = correctMatches(:,perm);
end

% Case N-1 matches -> we consider N-N geometry matches
% descMatches is a 1xNumFramesA vector
% geometryMatches is a NumFramesA x NumFramesB table

function [pairedMatches, correctMatches, sortedDistDescMatches] = ...
    LabelCorrectN21MatchesAndRecast(descMatches, geometryMatches, distDescMatches)
    
    % Next sentences should be redundant, since al descMatches entries
    % should be a frame index of B.
    
    idFramesA = find(descMatches ~= 0);
    pairedMatches = [idFramesA; descMatches(idFramesA)];
    correctMatches = zeros(1,size(descMatches,2));
    for i=1:size(descMatches,2) 
        if geometryMatches(i,descMatches(i))
            correctMatches(i) = 1;
        end
    end
        
    [sortedDistDescMatches,perm] = sort(distDescMatches,'ascend');  
    pairedMatches = pairedMatches(:, perm);
    correctMatches = correctMatches(:,perm);
end

% Case N-N matches -> we consider N-N geometry matches
% descMatches is a 2x(NumFramesA*NumFramesB) vector, detailing pairing
% (Ai,Bj)
% geometryMatches is a NumFramesA x NumFramesB table

function  [pairedMatches, correctMatches, sortedDistDescMatches] = ...
            LabelCorrectN2NMatchesAndRecast(descMatches,...
                                         geometryMatches,...
                                         distDescMatches)
                                     
    % Only correct matches has to be computed, since matches are already 
    % specified paired, and sorted according distDescMatches
    
    pairedMatches = descMatches;
    sortedDistDescMatches = distDescMatches;
    
    correctMatches = zeros(1,size(descMatches,2));
    for i=1:size(descMatches,2) 
        if geometryMatches(descMatches(1,i),descMatches(2,i))
            correctMatches(i) = 1;
        end
    end
end


% Cï¿½lcul de la precision - recall curve. Es va considerant un
% element mï¿½s cada vegada com a 'match correcte', i es com si
% anessim considerant cada vegada un threshold mï¿½s permisiu. Aquest
% nou element es valora si ï¿½s un true positive o false positive, i
% en funciï¿½ d'aixï¿½ es calcula el valor de precision i recall
        
% El total de correspondencies que en teoria s'haurien d'establir
% sï¿½n les que s'han generat a partir de l'error de solapament < 40%
% emprant la homografia GT entre imatges.


function [auc, aucGM, precision, recall, recallGM ] = ...
    PrecisionRecallComputation(correctMatches, numGTCorrespondences)
    
%     precision = [];
%     recall = [];
%     recallGM = [];
%         
%     % Inicialment tots els emparellaments els considerem negatius
%         
%     tp = 0;
%     fp = 0;
%     fn = sum(correctMatches);
%     tn = sum(~correctMatches);
%                 
%     for i=1:size(correctMatches,2)
%         if correctMatches(i)
%            tp = tp+1;
%            fn = fn-1;
%         else
%            fp = fp+1;
%            tn = tn-1;
%         end
%                         
%         % El quocient de recall hauria de ser sempre igual a sum(correctMatches)
%         recall(i) = tp/(tp+fn);
%         recallGM(i) = tp/numGTCorrespondences;
%         precision(i) = tp/(tp+fp); 
%     end
    
    % Versió vectoritzada del codi anterior
    
    tpv = cumsum(correctMatches);
    fnv = sum(correctMatches) - cumsum(correctMatches);
    fpv = cumsum(~correctMatches);
    tnv = sum(~correctMatches) - cumsum(~correctMatches);
    
    recall = tpv./(tpv+fnv);
    recallGM = tpv/numGTCorrespondences;
    precision = tpv./(tpv+fpv); 

    auc = trapz(recall,precision);
    aucGM = trapz(recallGM, precision);
end



