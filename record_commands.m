volume = SoundVolume;
disp(volume);
SoundVolume(1);
volume=SoundVolume;
info = audiodevinfo;
info = squeeze(struct2cell(info.input))';
info = info(:,[1 3]);
disp('Audioeingänge:')
disp(info)
id = input('ID des Audioeingangs (= Nr in rechter Spalte): ');
iterations = input("wie oft willst du den befehl aufnehmen?");
label = 'volume';
fs = 44100;
nbit = 24;
nch = 1;
recobj = audiorecorder(fs,nbit,nch,id);


offset=10;
for i = 1+offset:iterations+offset
    disp("next");
    recordblocking(recobj,1);
    x = getaudiodata(recobj);
    
    %wie speichern in anderem folder
    audiowrite(['recordings\',label,int2str(i),'.wav'],x,fs);
end