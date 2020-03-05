%Training data immagini ed etichette
load('gTruth.mat')
[imds,pxds] =pixelLabelTrainingData(gTruth);
%carico la rete 
net = alexnet;
layers = net.Layers;
%Modifico impostazioni net, cambio grandezza di inpute dell'immagine
imageSize = [480 640]; %Grandezza immagini
numClasses =3; % numeroClassi

tbl = countEachLabel(pxds);
%%Modifico primo strato  impostando la grandezza voluta
layers(1) = imageInputLayer([imageSize 3], 'Name', layers(1).Name,...
    'DataAugmentation', layers(1).DataAugmentation, ...
    'Normalization', layers(1).Normalization);

%recupero il peso dello stato 17 e creo un nuovo byas 



% fc6 is layers 17
idx = 17;
weights = layers(idx).Weights';
weights = reshape(weights, 6, 6, 256, 4096);
bias = reshape(layers(idx).Bias, 1, 1, []);

layers(idx) = convolution2dLayer(6, 4096, 'NumChannels', 256, 'Name', 'fc6');
layers(idx).Weights = weights;
layers(idx).Bias = bias;

% fc7 is layers 20
idx = 20;
weights = layers(idx).Weights';
weights = reshape(weights, 1, 1, 4096, 4096);
bias = reshape(layers(idx).Bias, 1, 1, []);

layers(idx) = convolution2dLayer(1, 4096, 'NumChannels', 4096, 'Name', 'fc7');
layers(idx).Weights = weights;
layers(idx).Bias = bias;


%Pad [100 100] per garantire che l'output della rete corrisponda all'immagine in ingresso. 
%(Ritaglia l'output della rete in modo che alla fine l'immagine
%in ingresso corrisponda all'area pertinente, in modo che abbia un margine. )
conv1 = layers(2);
conv1New = convolution2dLayer(conv1.FilterSize, conv1.NumFilters, ...
    'Stride', conv1.Stride, ...
    'Padding', [100 100], ...
    'NumChannels', conv1.NumChannels, ...
    'WeightLearnRateFactor', conv1.WeightLearnRateFactor, ...
    'WeightL2Factor', conv1.WeightL2Factor, ...
    'BiasLearnRateFactor', conv1.BiasLearnRateFactor, ...
    'BiasL2Factor', conv1.BiasL2Factor, ...
    'Name', conv1.Name);
conv1New.Weights = conv1.Weights;
conv1New.Bias = conv1.Bias;

layers(2) = conv1New;
%Rimuovo il livello di classificazione da AlexNet.
layers(end-2:end) = [];


%Creo up score La mappa delle caratteristiche ottenuta attraverso il livello 
%di convoluzione e pool ha una risoluzione ridotta, quindi 
%definisce un livello di convoluzione transpostionale per il upsampling.
upscore = transposedConv2dLayer(64, numClasses, ...
    'NumChannels', numClasses, 'Stride', 32, 'Name', 'upscore');

%Collego i layer creati.
layers = [
    layers
    convolution2dLayer(1, numClasses, 'Name', 'score_fr');
    upscore
    crop2dLayer('centercrop', 'Name', 'score')
    softmaxLayer('Name', 'softmax')
    pixelClassificationLayer('Name', 'pixelLabels')
    ];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'data', 'score/ref');

%Come accennato in precedenza, il numero totale di pixel in ogni classe è irregolare, 
%quindi il peso è basato sul numero totale di pixel.
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;


imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
pxLayer = pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)

lgraph = removeLayers(lgraph, 'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax' ,'labels');


figure, plot(lgraph)

%Preparo i dati in input 
options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'Plots','training-progress', ...
    'VerboseFrequency', 100);
%A causa del numero limitato di immagini disponibili per l'apprendimento,
%aumento il numero di immagini. Capovolgo l'immagine in orizzontale
%e ruoto in direzione X/Y nell'intervallo di +/- 10 pixel.
augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);
datasource = pixelLabelImageSource(imds,pxds,...
    'DataAugmentation',augmenter);
%Alleno la rete 
[net, info] = trainNetwork(datasource,lgraph,options);
%Stinghe per etichette
classes = [ "Globuli bianchi" , "Globuli rossi" , "Plasma"];
cmap = ColorMap();
%Leggo immagine test
I = imread('C:\Users\ladul\Desktop\Tirocinio\Madhloom\test\BloodImage_00382.jpg');
C = semanticseg(I, net);
%aggiungo etichette e stampo 
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
figure, imshowpair(I, B, 'montage')
pixelLabelColorbar(cmap, classes);

%%Test dati 
