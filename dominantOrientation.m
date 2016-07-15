function angle = dominantOrientation(img)

[imgMag, imgDir] = gradmag(img,1);

bin_nb = 90;
his = zeros(1, bin_nb);
imgDir = floor((bin_nb-1)*(imgDir+pi)/(2*pi))+1;

[h w] = size(img);
radius = size(img,1)/5;

r2 = radius^2;
half = floor(h/2);

for y=1:h
    for x=1:w
        mag = imgMag(y,x);
        ori = imgDir(y,x);
        if (x-half)^2 + (y-half)^2 < r2
            his(ori) = his(ori) + mag;
        end
    end
end

 % figure(3);stem(4*[1:bin_nb],his);

[m in] = max(his);

angle = (2*pi*in)/bin_nb;
end

function [imgMag, imgDir] = gradmag(img, sigma)

x = [floor(-3.0*sigma+0.5):floor(3.0*sigma+0.5)];
G = exp(-x.^2/(2*sigma^2));
G = G/sum(sum(G));
D = -2*(x.*exp(-x.^2/(2*sigma^2)))/(sqrt(2*pi)*sigma^3);

img1  = conv2(img , D , 'same');
imgDx = conv2(img1, G', 'same');
img2  = conv2(img , D', 'same');
imgDy = conv2(img2, G , 'same');

imgMag = sqrt((imgDx.^2) + (imgDy.^2));
imgDir = atan2(imgDy,imgDx);

end