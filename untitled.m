%Carico le immagini etichettate 
load('labelsglobuli.mat');
%Preparo il mio training
trainingData = objectDetectorTrainingData(gTruth);
%Costruisco il mio detector
detector = trainACFObjectDetector(trainingData);

%Carico un immagine
img = imread('C:\Users\ladul\Desktop\Tirocinio\Madhloom\img\BloodImage_00300.jpg');
%La do in pasto al detectro e separo le etichette dal punteggio 
[bboxes, scores] = detect(detector,I,'Threshold',0.1);
[~,idx] =  max(scores); 




%Stampo i risutlati a video 

annotation = detector.ModelName;

I = insertObjectAnnotation(img,'rectangle',bboxes(idx,:),annotation);

figure 
imshow(I)