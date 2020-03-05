function cmap = ColorMap()
%
% Define the colormap used by  dataset.

cmap = [
    60 40 222   % WBC
    128 0 0     % RBC
    064 128 064  %Plasma
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end
