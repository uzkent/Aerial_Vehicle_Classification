%This function converts HSI chips to individual images by sampling one
%image from every 10 bands
Train = load('dirsigSamples.mat');
trainSamples = Train.trainingSample;
trainLabels = Train.label;
clear Train;

% Update the Labels as 0,1,2,3,4,5
lb = unique(trainLabels);
temp = trainLabels;
for i = 1:size(lb,2)
    ind = find(trainLabels==lb(i));
    temp(ind) = i - 1;
end
trainLabels = temp;
clear temp;

%Find PC
X = [];
for i = 1:1000
    %Flatten the Matrix
    index = randi([size(trainingSample,1)]);
    trainSamples{i} = NoiseAdd(trainingSample{i},0.1*randi([6 9]));
    mn = reshape(trainingSample{i},size(trainingSample{i},1)*...
    size(trainingSample{i},2),61);
    %Concatanate
    X = [X;mn];
end
[coeff] = pca(X);

%Go through each chip and sample images
counter = 1;
save_dir = [];
for i = 1:size(trainingSample,2)
    %Apply the PCA to the HSI Image
    temp = reshape(NoiseAdd(trainingSample{i},0.1*randi([6 9])),64*64,61);
    img(:,:,1) = reshape(sum(coeff(:,1)' .* temp,2),64,64,1);
    img(:,:,2) = reshape(sum(coeff(:,2)' .* temp,2),64,64,1);
    img(:,:,3) = reshape(sum(coeff(:,3)' .* temp,2),64,64,1);
    %Save it
    save_folder = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/HSI_Classifier/Images/';
    save_dir = [save_dir;save_folder 'image_' sprintf('%05d',counter) '.png'];
    img = 255 * img ./ max(max(max(img))); %normalize the band
    imwrite(uint8(img),save_dir(end,:)); %save it
    counter = counter + 1; %update the counter
end
for i = 1:size(label,2)
    GT(i,:) = [save_dir(i,:) ' ' sprintf('%02d',label(i))];
end
lbs = unique(label);
temp = label;
for i = 1:size(lbs,2)
    inds = find(label == lbs(i));
    temp(inds) = i - 1;
end
%save ground truth text file
dlmwrite([save_folder 'train_HSI.txt'],GT,'delimiter','');