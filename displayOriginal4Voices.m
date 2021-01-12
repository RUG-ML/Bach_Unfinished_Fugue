% play solutions
addpath("TXT_files\F.txt");

% load the data: 2 voices in columns
load TXT_files\F.txt -ascii;
%%%%%%%% transform a voice into a soundvector that can be played by Matlab 

% choose a voice 
chosenVoice = 1;
voice = F(:,chosenVoice);
% plot it to get the Bach feeling 
figure(1); plot(voice);

symbolicLength = length(voice);
% find minimal key Number
minKeyNr = 1000;
for n = 1:symbolicLength
    if voice(n) < minKeyNr & voice(n) ~= 0
        minKeyNr = voice(n);
    end
end
disp(minKeyNr);

% set parameters for playing sound
baseFreq = 440;% set base frequency (Hz) of lowest note "minKeyNr" in voice
sampleRate = 10000; % samples per second
durationPerSymbol = 1/20; % in seconds. A "symbol" here means one entry in the voice vector
ticksPerSymbol = floor(sampleRate * durationPerSymbol);

% transform to soundvector
soundvector1 = zeros(symbolicLength * ticksPerSymbol,1);
currentSymbol = voice(1); startSymbolIndex = 1;
for n = 1:symbolicLength
    if voice(n) ~= currentSymbol
        stopSymbolIndex = n-1;
        coveredSoundVectorIndices = ...
            (startSymbolIndex -1)* ticksPerSymbol + 1:...
            stopSymbolIndex * ticksPerSymbol ;
        toneLength = length(coveredSoundVectorIndices);
        frequency = baseFreq * 2^((currentSymbol - minKeyNr)/12 );        
        toneVector = zeros(toneLength,1);
        for t = 1:toneLength
            toneVector(t,1) = sin(2 * pi * frequency * t / sampleRate);
        end
        soundvector1(coveredSoundVectorIndices,1) = toneVector;
        currentSymbol = voice(n);
        startSymbolIndex = n;    
    end    
end


% transform another voice into another soundvector
chosenVoice = 2;
voice = F(:,chosenVoice);
figure(2); plot(voice);
soundvector2 = zeros(symbolicLength * ticksPerSymbol,1);
currentSymbol = voice(1); startSymbolIndex = 1;
for n = 1:symbolicLength
    if voice(n) ~= currentSymbol
        stopSymbolIndex = n-1;
        coveredSoundVectorIndices = ...
            (startSymbolIndex -1)* ticksPerSymbol + 1:...
            stopSymbolIndex * ticksPerSymbol ;
        toneLength = length(coveredSoundVectorIndices);
        frequency = baseFreq * 2^((currentSymbol - minKeyNr)/12 );        
        toneVector = zeros(toneLength,1);
        for t = 1:toneLength
            toneVector(t,1) = sin(2 * pi * frequency * t / sampleRate);
        end
        soundvector2(coveredSoundVectorIndices,1) = toneVector;
        currentSymbol = voice(n);
        startSymbolIndex = n;    
    end    
end

chosenVoice = 3;
voice = F(:,chosenVoice);
figure(3); plot(voice);
soundvector3 = zeros(symbolicLength * ticksPerSymbol,1);
currentSymbol = voice(1); startSymbolIndex = 1;
for n = 1:symbolicLength
    if voice(n) ~= currentSymbol
        stopSymbolIndex = n-1;
        coveredSoundVectorIndices = ...
            (startSymbolIndex -1)* ticksPerSymbol + 1:...
            stopSymbolIndex * ticksPerSymbol ;
        toneLength = length(coveredSoundVectorIndices);
        frequency = baseFreq * 2^((currentSymbol - minKeyNr)/12 );        
        toneVector = zeros(toneLength,1);
        for t = 1:toneLength
            toneVector(t,1) = sin(2 * pi * frequency * t / sampleRate);
        end
        soundvector3(coveredSoundVectorIndices,1) = toneVector;
        currentSymbol = voice(n);
        startSymbolIndex = n;    
    end    
end

chosenVoice = 4;
voice = F(:,chosenVoice);
figure(4); plot(voice);
soundvector4 = zeros(symbolicLength * ticksPerSymbol,1);
currentSymbol = voice(1); startSymbolIndex = 1;
for n = 1:symbolicLength
    if voice(n) ~= currentSymbol
        stopSymbolIndex = n-1;
        coveredSoundVectorIndices = ...
            (startSymbolIndex -1)* ticksPerSymbol + 1:...
            stopSymbolIndex * ticksPerSymbol ;
        toneLength = length(coveredSoundVectorIndices);
        frequency = baseFreq * 2^((currentSymbol - minKeyNr)/12 );        
        toneVector = zeros(toneLength,1);
        for t = 1:toneLength
            toneVector(t,1) = sin(2 * pi * frequency * t / sampleRate);
        end
        soundvector4(coveredSoundVectorIndices,1) = toneVector;
        currentSymbol = voice(n);
        startSymbolIndex = n;    
    end    
end

% add the two soundvectors to get a 2-voice score
soundvector = (soundvector1 + soundvector2 + soundvector3 + soundvector4)/ 4;
% sound(soundvector, 10000);
audiowrite("Bach.wav", soundvector, 10000);
