%% THIS FUNCTION READ ENVI IMAGES AND SAVES THEM in .MAT FORMAT
images = dir('/Volumes/Burak_HardDrive/Moving_Platform_HSI_NoTrees_2/');
images(1:2) = [];
images(1:69) = [];
home_dir = '/Volumes/Burak_HardDrive/Moving_Platform_HSI_NoTrees_2/Image_';
for i = 1:157

    [img,~] = enviread(images(i*2-1).name,images(i*2).name);
    
    save_image(home_dir,img,i);

end