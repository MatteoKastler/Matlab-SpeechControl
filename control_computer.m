% %deep learning tutorial
% %https://de.mathworks.com/help/deeplearning/ug/deep-learning-speech-recognition.html#mw_rtc_DeepLearningSpeechRecognitionExample_M_DCA98732
%
% %press buttons
% %https://de.mathworks.com/matlabcentral/answers/479760-how-to-programmatically-press-a-key
% 
% 
% rng default
% 
% downloadFolder = matlab.internal.examples.downloadSupportFile("audio","google_speech.zip");
% dataFolder = "C:\Users\matte\Desktop\HCIN\final_project";
% %unzip(downloadFolder,dataFolder)
% dataset = fullfile(dataFolder,"google_speech");
% 
% 
% %training Datastore
% ads_train = audioDatastore(fullfile(dataset,"train"),...
%     IncludeSubfolders=true, ...
%     FileExtensions=".wav", ...
%     LabelSource="foldernames"); %namen als label benutzen
% 
% %eigene dateien aufnehmen und in google datensatz noch einfügen
% %"brightness", "volume"
% commands = categorical(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine","up", "down"]);
% background = categorical("background");
% 
% isCommand = ismember(ads_train.Labels,commands);
% isBackground = ismember(ads_train.Labels,background);
% isUnknown = ~(isCommand|isBackground);
% 
% includeFraction = 0.2; % Fraction of unknowns to include.
% idx = find(isUnknown);
% idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
% isUnknown(idx) = false;
% 
% ads_train.Labels(isUnknown) = categorical("unknown");
% 
% adsTrain = subset(ads_train,isCommand|isUnknown|isBackground);
% adsTrain.Labels = removecats(adsTrain.Labels);
% 
% 
% %validation Datastore
% ads_val = audioDatastore(fullfile(dataset,"validation"), ...
%    IncludeSubfolders=true, ...
%    FileExtensions=".wav", ...
%    LabelSource="foldernames");
% 
% isCommand = ismember(ads_val.Labels,commands);
% isBackground = ismember(ads_val.Labels,background);
% isUnknown = ~(isCommand|isBackground);
% 
% includeFraction = 0.2; % Fraction of unknowns to include.
% idx = find(isUnknown);
% idx = idx(randperm(numel(idx),round((1-includeFraction)*sum(isUnknown))));
% isUnknown(idx) = false;
% 
% ads_val.Labels(isUnknown) = categorical("unknown");
% 
% adsValidation = subset(ads_val,isCommand|isUnknown|isBackground);
% adsValidation.Labels = removecats(adsValidation.Labels);
% disp("created Datastores")
% 
% %reduce dataset to speed up
% numUniqueLabels = numel(unique(adsTrain.Labels));
%     % Reduce the dataset by a factor
%     adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / numUniqueLabels / 2));
%     adsValidation = splitEachLabel(adsValidation,round(numel(adsValidation.Files) / numUniqueLabels / 2));
% 
%set parameters for feature extraction
fs = 16e3; % Known sample rate of the data set.
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;

FFTLength = 512;
numBands = 50;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

%extract features
afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    FFTLength=FFTLength, ...
    Window=hann(frameSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);
setExtractorParameters(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);
% 
% %TRAINING DATA
% %pad to length, extract features, apply logarithm
% transform1 = transform(adsTrain,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
% transform2 = transform(transform1,@(x)extract(afe,x));
% transform3 = transform(transform2,@(x){log10(x+1e-6)});
% 
% %get data
% XTrain = readall(transform3);
% disp("spectograms extracted: ");
% disp(numel(XTrain));
% 
% %convert data to 4dim array
% [numHops,numBands,numChannels] = size(XTrain{1});
% XTrain = cat(4,XTrain{:});
% 
% 
% %VALIDATION DATA
% transform1 = transform(adsValidation,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
% transform2 = transform(transform1,@(x)extract(afe,x));
% transform3 = transform(transform2,@(x){log10(x+1e-6)});
% 
% %get data
% XValidation = readall(transform3);
% XValidation = cat(4,XValidation{:});
% 
% %isolate target labels
% TTrain = adsTrain.Labels;
% TValidation = adsValidation.Labels;
% 
% 
% 
% %NETWORK ARCHITECTURE
% classes = categories(TTrain);
% classWeights = 1./countcats(TTrain);
% classWeights = classWeights'/mean(classWeights);
% numClasses = numel(classes);
% 
% timePoolSize = ceil(numHops/8);
% 
% dropoutProb = 0.2;
% numF = 12;
% layers = [
%     imageInputLayer([numHops,afe.FeatureVectorLength])
%     
%     convolution2dLayer(3,numF,Padding="same")
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(3,Stride=2,Padding="same")
%     
%     convolution2dLayer(3,2*numF,Padding="same")
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(3,Stride=2,Padding="same")
%     
%     convolution2dLayer(3,4*numF,Padding="same")
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(3,Stride=2,Padding="same")
%     
%     convolution2dLayer(3,4*numF,Padding="same")
%     batchNormalizationLayer
%     reluLayer
% 
%     convolution2dLayer(3,4*numF,Padding="same")
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer([timePoolSize,1])
%     dropoutLayer(dropoutProb)
% 
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer(Classes=classes,ClassWeights=classWeights)];
% 
% %Training options
% miniBatchSize = 128;
% validationFrequency = floor(numel(TTrain)/miniBatchSize);
% options = trainingOptions("adam", ...
%     InitialLearnRate=3e-4, ...
%     MaxEpochs=20, ...
%     MiniBatchSize=miniBatchSize, ...
%     Shuffle="every-epoch", ...
%     Plots="training-progress", ...
%     Verbose=false, ...
%     ValidationData={XValidation,TValidation}, ...
%     ValidationFrequency=validationFrequency);
% 
% %train the network
% trainedNet = trainNetwork(XTrain,TTrain,layers,options);
% save("trainedNet_final.mat", "trainedNet");

%EVALUATE NETWORK
%load trained network
load("trainedNet_final.mat");

%robot for pressing buttons
import java.awt.Robot;
import java.awt.event.*;
robot = Robot();

%classify input
info = audiodevinfo;
info = squeeze(struct2cell(info.input))';
info = info(:,[1 3]);
disp('Audioeingänge:')
disp(info)
id = input('ID des Audioeingangs (= Nr in rechter Spalte): ');

nbit = 24;
nch = 1;

recobj = audiorecorder(fs,nbit,nch,id);

for i=0:5
    disp("jetzt");
    recordblocking(recobj,1);
    
    %abspeichern und mit datastore wieder einlesen damit aussieht wie adstrain
    to_classify = getaudiodata(recobj);
    audiowrite(fullfile("recordings\"+"spoken_temp.wav"), to_classify,fs);
    
    ads_spoken = audioDatastore("recordings\" + "spoken_temp.wav",...
      IncludeSubfolders=false, ...
      FileExtensions=".wav");
    
     transform1 = transform(ads_spoken,@(x)[zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)]);
     transform2 = transform(transform1,@(x)extract(afe,x));
     transform3 = transform(transform2,@(x){log10(x+1e-6)});
    
    Xspoken = readall(transform3);
    disp("spoken data: ");
    disp(numel(Xspoken));


    %convert data to 4dim array
    [numHops,numBands,numChannels] = size(Xspoken{1});
    Xspoken = cat(4,Xspoken{:});
    
    result = classify(trainedNet,Xspoken);
    disp(result);
    if result(1,1) == 'down'
       robot.keyPress(KeyEvent.VK_DOWN);
       robot.keyRelease(KeyEvent.VK_DOWN);

        robot.keyPress(KeyEvent.VK_DOWN);
       robot.keyRelease(KeyEvent.VK_DOWN);

        robot.keyPress(KeyEvent.VK_DOWN);
       robot.keyRelease(KeyEvent.VK_DOWN);

        robot.keyPress(KeyEvent.VK_DOWN);
       robot.keyRelease(KeyEvent.VK_DOWN);

        robot.keyPress(KeyEvent.VK_DOWN);
       robot.keyRelease(KeyEvent.VK_DOWN);
    end
    
    if result(1,1) == 'up'
       robot.keyPress(KeyEvent.VK_UP);
       robot.keyRelease(KeyEvent.VK_UP);

       robot.keyPress(KeyEvent.VK_UP);
       robot.keyRelease(KeyEvent.VK_UP);

       robot.keyPress(KeyEvent.VK_UP);
       robot.keyRelease(KeyEvent.VK_UP);

       robot.keyPress(KeyEvent.VK_UP);
       robot.keyRelease(KeyEvent.VK_UP);

       robot.keyPress(KeyEvent.VK_UP);
       robot.keyRelease(KeyEvent.VK_UP);
    end

    if result(1,1) == 'one'
        robot.keyPress(KeyEvent.VK_1);
        robot.keyRelease(KeyEvent.VK_1);
    end

    if result(1,1) == 'two'
        robot.keyPress(KeyEvent.VK_2);
        robot.keyRelease(KeyEvent.VK_2);
    end

    if result(1,1) == 'three'
        robot.keyPress(KeyEvent.VK_3);
        robot.keyRelease(KeyEvent.VK_3);
    end

    if result(1,1) == 'four'
        robot.keyPress(KeyEvent.VK_4);
        robot.keyRelease(KeyEvent.VK_4);
    end

    if result(1,1) == 'five'
        robot.keyPress(KeyEvent.VK_5);
        robot.keyRelease(KeyEvent.VK_5);
    end

    if result(1,1) == 'six'
        robot.keyPress(KeyEvent.VK_6);
        robot.keyRelease(KeyEvent.VK_6);
    end
    pause(3);
end


%calculate validity
% YValidation = classify(trainedNet,XValidation);
% validationError = mean(YValidation ~= TValidation);
% 
% YTrain = classify(trainedNet,XTrain);
% trainError = mean(YTrain ~= TTrain);
% 
% disp(["Training error: " + trainError*100 + "%";"Validation error: " + validationError*100 + "%"])
% 
% %calculate confusion matrix
% figure(Units="normalized",Position=[0.2,0.2,0.5,0.5]);
% cm = confusionchart(TValidation,YValidation, ...
%     Title="Confusion Matrix for Validation Data", ...
%     ColumnSummary="column-normalized",RowSummary="row-normalized");
% sortClasses(cm,[commands,"unknown"]);
% 