clear all
close all

figure(1)
x = 0:0.1:1;
y = 0:0.1:1;

load('harris_oxford.mat')
A = auc2;

sift = 1;
siam = 2;
simo = 3;
pnnet = 4;
matchnet = 5;
siam2stream = 6;
pnnet2 = 7;

desc1 = sift;
desc2 = pnnet;

num_datasets = size(A,1);
num_images = size(A,2);
num_descriptors = size(A,3);

X = [];
Y = [];

for i=1:num_datasets
    a = reshape(A(i,:,:), [num_images num_descriptors]);
    X = [X a(desc1,:)];
    Y = [Y a(desc2,:)];
end

plot(X,Y,'ro',x,y,'k--','linewidth',3)
xlabel(['AP with SIFT'],'fontsize',15);
ylabel(['AP with PN-Net'],'fontsize',15);
title('SymBench: PN-Net vs SIFT','fontsize',20)