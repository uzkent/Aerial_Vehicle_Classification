% This script visualizes the samples from training dataset
% It picks negative and positive samples randomly and shows them in a
% figure
% Dirsig training data directory
clear;
close all;
dirsig_dir = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/Binary_Classifier/Images/';
wami_dir = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/Binary_Classifier/Images_Val/';

% Read the Dirsig Training Data Information
fid = fopen([dirsig_dir 'train.txt']);
dirsig_info = textscan(fid,'%s %d');

% Read the WAMI Training Data Information
fid = fopen([wami_dir 'val.txt']);
wami_info = textscan(fid,'%s %d');

% Sample N number of images from DIRSIG
N = 144;
number_dirsig_images = size(dirsig_info{1},1);
positive_all_indexes = find(dirsig_info{2}==1);
negative_all_indexes = find(dirsig_info{2}==0);
positive_indexes = randi(size(positive_all_indexes,1),[1 N]);
negative_indexes = randi(size(negative_all_indexes,1),[1 N]);

% Sample N number of images from WAMI
number_wami_images = size(wami_info{1},1);
positive_all_windexes = find(wami_info{2}==1);
negative_all_windexes = find(wami_info{2}==0);
positive_windexes = randi(size(positive_all_windexes,1),[1 N]);
negative_windexes = randi(size(negative_all_windexes,1),[1 N]);

% Display the Dirsig positives
row = sqrt(N);
[ha, pos] = tight_subplot(row,row,[.003 -0.2],[.05 .005],[.05 .005]);
counter = 1;
figure(1)
for ii = 1:N
    axes(ha(ii));
    ind = positive_all_indexes(positive_indexes(counter));
    img = imread([dirsig_dir 'image_' sprintf('%05d',ind) '.png']);
    imshow(img);
    counter = counter + 1;
end

% Display the wami positives
row = sqrt(N);
figure(2)
[ha, pos] = tight_subplot(row,row,[.003 -0.2],[.05 .005],[.05 .005]);
counter = 1;
for ii = 1:N
    axes(ha(ii));
    ind = positive_all_windexes(positive_windexes(counter));
    img = imread([wami_dir 'image_' sprintf('%05d',ind) '.png']);
    imshow(img);
    counter = counter + 1;
end

% Display the Dirsig negatives
row = sqrt(N);
figure(3)
[ha, pos] = tight_subplot(row,row,[.003 -0.2],[.05 .005],[.05 .005]);
counter = 1;
for ii = 1:N
    axes(ha(ii));
    ind = negative_all_indexes(negative_indexes(counter));
    img = imread([dirsig_dir 'image_' sprintf('%05d',ind) '.png']);
    imshow(img);
    counter = counter + 1;
end

% Display the wami negatives
row = sqrt(N);
figure(4)
[ha, pos] = tight_subplot(row,row,[.003 -0.2],[.05 .005],[.05 .005]);
counter = 1;
for ii = 1:N
    axes(ha(ii));
    ind = negative_all_windexes(negative_windexes(counter));
    img = imread([wami_dir 'image_' sprintf('%05d',ind) '.png']);
    imshow(img);
    counter = counter + 1;
end