function patches = normalize_regions(imgfile, featurefile)

img = imread(imgfile);

%if color image
if size(size(img),2)==3 
  img=rgb2gray(img);
end

% figure(1);colormap gray;
% figure(2);colormap gray;

[feat nb dim] = loadFeatures(featurefile);

%fprintf(1,'press a key to display next patch...');

%Scaling factor measurement region/distinguished region
N = 41;
scaling = 3;
patches = zeros(N,N,nb);

for f=1:nb
    angle = 0;
    patch = normalizePatch(img, feat(f,1),feat(f,2),feat(f,3),feat(f,4),feat(f,5),angle,scaling,N);
    angle = dominantOrientation(patch);

    patch_norm = normalizePatch(img, feat(f,1),feat(f,2),feat(f,3),feat(f,4),feat(f,5),angle,scaling,N);
    patches(:,:,f) = mat2gray(patch_norm);

%     figure(1);imagesc(img);axis image;
%     drawellipse(feat(f,1),feat(f,2), feat(f,3), feat(f,4),feat(f,5),scaling,'-y');
%     figure(2);imagesc(patches(:,:,f));axis image;
%    pause;

end

end

%%%%%%%DISPLAY ELLIPSE
function drawellipse(x,y,a,b,c,scaling,col)
hold on;
[v e]=eig([a b;b c]);

l1=1/sqrt(e(1));
l2=1/sqrt(e(4));

alpha=atan2(v(4),v(3));
t = 0:pi/50:2*pi;
yt=scaling*(l2*sin(t));
xt=scaling*(l1*cos(t));

p=[xt;yt];
R=[cos(alpha) sin(alpha);-sin(alpha) cos(alpha)];
pt=R*p;
plot(pt(2,:)+x,pt(1,:)+y,col,'LineWidth',1);
%set(gca,'Position',[0 0 1 1]);
hold off;
end

%%%%%%%load Features
function [feat nb dim]=loadFeatures(file)
fid = fopen(file, 'r');
dim=fscanf(fid, '%f',1);
if dim==1
dim=0;
end
nb=fscanf(fid, '%d',1);
feat = fscanf(fid, '%f', [5+dim, inf]);
feat=feat';
fclose(fid);
end