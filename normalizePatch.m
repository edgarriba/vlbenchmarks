function imout = normalizePatch(img,x,y,a,b,c,angle,scaling,normalized_patch_size)

A = [a b;b c];
A = A^(-0.5);  %inverse square root
sc = 2*scaling/normalized_patch_size; % 2 because patch_size is a diameter (not radius)

A = A*sc; %Increasing the size of distinguished region
R = [cos(angle), -sin(angle);...
     sin(angle),  cos(angle)];
RA = A*R; %normalizing with dominant angle

imout = zeros(normalized_patch_size, normalized_patch_size);
[h w]=size(img);
half_size=floor(normalized_patch_size/2);

%compute the transformed region with bilinear interpolation
for j=-half_size:half_size
    for i=-half_size:half_size

        pt = RA*[i j]';
        pts = floor(pt);
        xt = real(int32(pts(1)));
        yt = real(int32(pts(2)));
        dx = real(pt(1)-pts(1));
        dy = real(pt(2)-pts(2));

        if x+xt > 0 && x+xt+1 < w && y+yt > 0 && y+yt+1 < h
            imout(half_size+1+j, half_size+1+i) = img(y+yt,x+xt)*(1-dx)*(1-dy) +...
                                                  img(y+yt+1,x+xt)*(1-dx)*(dy) +...
                                                  img(y+yt,x+xt+1)*(dx)*(1-dy) +...
                                                  img(y+yt+1,x+xt+1)*(dx)*(dy); 
        else
            imout(half_size+1+j, half_size+1+i) = 0;
        end

    end
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
