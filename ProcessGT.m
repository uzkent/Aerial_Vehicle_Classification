%% THIS FUNCTION READ ENVI IMAGES AND SAVES THEM in .MAT FORMAT
home_dir{1} = '/Volumes/Seagate Backup Plus Drive/Moving_Platform_CNN_Training/Training/';
home_dir{2} = '/Volumes/Seagate Backup Plus Drive/Moving_Platform_CNN_Training/Training2/';
sampleTimes{1} = '10am';
sampleTimes{2} = '1pm';
sampleTimes{3} = '3pm';
sampleTimes{4} = '5pm';
sampleTimes{5} = '11am';
sampleTimes{6} = '2pm';
sampleTimes{7} = '4pm';


%First Process Ground Truths
counter = 1;   % Sample Counter
chipSize = 32; % Pixels
for d = 1:size(home_dir,2)
    for i = 1:size(sampleTimes,2)

        %Find Images in the current time set
        images = dir([home_dir{d} sampleTimes{i} '/GT_InstIndex']);
        images(1:2) = [];
        samples = dir([home_dir{d} sampleTimes{i} '/Images']);
        samples(1:2) = [];

        %Read GT Images and Process Them
        for j = 2:2:size(images,1)
            img = enviread([home_dir{d} sampleTimes{i} '/GT_InstIndex/' images(j-1).name]);
            imgLabel = enviread([home_dir{d} sampleTimes{i} '/GT_MatIndex/' images(j-1).name]);
            %Refine the Ground Truth Mask to Exclude non-vehicle objects
            img = img(:,:,1);
            img(img<5000) = 0;
            img(img>6000) = 0;
            %Apply Shadow Mask
            sMask = imgLabel(:,:,11);
            sMask(sMask>0) = 1;
            sMask(sMask==0) = 2;
            sMask = sMask - 1;
            img = img .* sMask;
            %Find Connected Components and Remove Heavily Occluded Ones
            CC = bwconncomp(img);
            for n = 1:CC.NumObjects
                if (size(CC.PixelIdxList{n},1) < 30)
                    %Remove them from the GT
                    img(CC.PixelIdxList{n}) = 0;
                end
            end
            temp = img;
            temp(temp~=0)=1;
            imgLabel = temp .* imgLabel(:,:,1); % Refine the Label Map

            % Assign Class
            % img(:,:,2) = 1;

            %Collect Training Samples from The Current Image
            s  = regionprops(img,'centroid');
            Image = enviread([home_dir{d} sampleTimes{i} '/Images/' samples(j-1).name]);
            validSamples = 0;
            for n = 1:size(s,1)
                if isnan(s(n).Centroid(1)) == 0
                    X = round(s(n).Centroid(1)) - chipSize:round(s(n).Centroid(1)) + (chipSize-1);
                    Y = round(s(n).Centroid(2)) - chipSize:round(s(n).Centroid(2)) + (chipSize-1);
                    if ((X(1) < 1) || (Y(1) < 1)) || ((X(end) > 1500) || (Y(end) > 1500))
                        continue;
                    end
                    trainingSample{counter} = Image(Y,X,:);
                    %Find the Label of th Sample
                    temp = imgLabel(Y,X);
                    uniqueIDs = unique(temp);
                    occurs = [uniqueIDs,histc(temp(:),uniqueIDs)];
                    occurs(1,:) = [];
                    [number,ind] = max(occurs(:,2));
                    label(counter) = occurs(ind,1);  %Pick the Label                
                    counter = counter + 1; % Update the Counter for Training Samples
                    validSamples = validSamples + 1;
                end
            end

            %Collect Negative Samples from the Image
            img(img~=0) = 1;
            [zeroIndx,zeroIndy] = find(img==0); %Find indices of zeros
            tempCounter = counter;
            while(counter < tempCounter+validSamples)
                ri = randi(size(zeroIndx,1));
                X = round(zeroIndx(ri)) - chipSize:round(zeroIndx(ri)) + (chipSize-1);
                Y = round(zeroIndy(ri)) - chipSize:round(zeroIndy(ri)) + (chipSize-1);
                if ((X(1) < 1) || (Y(1) < 1)) || ((X(end) > 1500) || (Y(end) > 1500))
                    continue;
                end
                if (sum(sum(img(Y,X))) == 0) || (sum(sum(img(Y,X))) >  20)
                    continue;
                end
                trainingSample{counter} = Image(Y,X,:);
                label(counter) = 0;  %Pick the Label                
                counter=counter+1;
            end        
        end   
    end
end
