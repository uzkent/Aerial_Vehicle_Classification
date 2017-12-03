%This function converts HSI chips to individual images by sampling one
%image from every 10 bands
Train = load('dirsigSamples.mat');
trainSamples = Train.trainingSample;
trainLabels = Train.label;
trainLabels(trainLabels>0)=1;
clear Train;
%Go through each chip and sample images
counter = 1;
save_dir = [];
for i = 1:size(trainSamples,2)
    for j = 1:11
        %Sample the band
        img = trainSamples{i}(:,:,(j-1)*5+5);
        img = NoiseAdd(img,0.1*randi([6 9]));
        %Save it
        save_folder = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/Binary_Classifier/Images/';
        save_dir = [save_dir;save_folder 'image_' sprintf('%05d',counter) '.png'];
        img = 255 * img ./ max(max(img)); % normalize the band
        imwrite(uint8(img),save_dir(end,:)); % save it
        labels(counter) = trainLabels(i);
        counter = counter + 1; %update the counter
    end
end
for i = 1:size(labels,2)
    GT(i,:) = [save_dir(i,:) ' ' num2str(labels(i))];
end
%save ground truth text file
dlmwrite([save_folder 'train.txt'],GT,'delimiter','');