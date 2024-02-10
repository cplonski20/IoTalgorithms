   # Working Code


   
def duet_source_separation(mic_data_folder, NUM_SOURCES):
    """DUET source separation algorithm. Write your code here.

    Args:
        mic_data_folder: name of folder (without a trailing slash) containing 
                         two mic datafiles `0.wav` and `1.wav`.

    Returns:
        NUM_SOURCES * recording_length numpy array, where NUM_SOURCES is the number of sources,
        and recording_length is the original length of the recording (in number of samples)

    """
    fs1, x1 = scipy.io.wavfile.read(mic_data_folder + '/0.wav')
    fs2, x2 = scipy.io.wavfile.read(mic_data_folder+ '/1.wav')

    nper = 512
    nover = 100
    toReturn = np.zeros((NUM_SOURCES,len(x1)))

    freqs1, times1, spec1 = signal.stft(x1,fs1, nperseg=nper, noverlap= nover) #nperseg default 256 noverlap 
    freqs2, times2, spec2 = signal.stft(x2,fs2, nperseg= nper, noverlap= nover)
    angs = []
    mags = []
    indices = [] # (frequency,time)

    for  j in range(0,times1.shape[0]): 
        for i in range(1,freqs1.shape[0]//2):
            if np.abs(spec1[i][j]) >  .1 and np.abs(spec2[i][j]) > .1:
                temp = spec2[i][j]/spec1[i][j]
                mags.append(np.abs(temp))
                # angs.append(np.angle(temp))
                angs.append(np.arctan2(temp.imag, temp.real)/(freqs1[i]))
                indices.append((i,j))


    angs = np.array(angs).reshape(-1,1)
    # mags = np.array(mags).reshape(-1,1)

    # toClass = np.column_stack((angs,mags))
    toClass = angs
    classifier = KMeans(n_clusters= NUM_SOURCES)
    classed = classifier.fit_predict(X = toClass)
    # classifier = GaussianMixture(n_components=NUM_SOURCES)
    # classed = classifier.fit_predict(X = toClass)

    for i in range(len(classed-10) //10): # smoothing the classifications
        classed[i*10:i*10 + 10] = np.round(np.mean(classed[i*10:i*10 + 10]))


    indices = np.array(indices)

    #good stuff i think from here on
    sourceindeces = []
    for i in range(NUM_SOURCES):
        sourceindeces.append(indices[classed == i])



    toinverse = []
    for source in sourceindeces:
        temp = np.zeros_like(spec1)
        for idx in source:
            temp[idx[0]][idx[1]] = spec1[idx[0], idx[1]]
        toinverse.append(temp)

    for i in range(NUM_SOURCES):
        toReturn[i,:] = scipy.signal.istft(toinverse[i], fs1, nperseg=nper, noverlap=nover)[1][:len(x1)]

    return 2**15 * toReturn.astype('int16')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    fs1, x1 = scipy.io.wavfile.read(mic_data_folder + '/0.wav')
    fs2, x2 = scipy.io.wavfile.read(mic_data_folder+ '/1.wav')

    if(NUM_SOURCES == 1):
        return np.array([x1])

    nper = 255
    nover = 0
    toReturn = np.zeros((NUM_SOURCES,len(x1)))

    freqs1, times1, spec1 = signal.stft(x1,fs1, nperseg=nper, noverlap= nover) #nperseg default 256 noverlap 
    freqs2, times2, spec2 = signal.stft(x2,fs2,nperseg= nper, noverlap= nover)
    angs = []
    mags = []
    indices = [] # (frequency,time)

    for  j in range(times1.shape[0]): 
        for i in range(freqs1.shape[0]):
            if np.abs(spec1[i][j]) >  1 and np.abs(spec2[i][j]) > 1:
                if(freqs1[i]%(2*np.pi) == 0 ):
                    continue
                temp = spec2[i][j]/spec1[i][j]
                mags.append(np.abs(temp))
                # angs.append(np.angle(temp)/freqs1[i])
                angs.append(np.arctan2(temp.imag, temp.real)/(freqs1[i]%(2*np.pi)))
                indices.append((i,j))


    angs = np.array(angs).reshape(-1,1)
    mags = np.array(mags).reshape(-1,1)

    toClass = np.column_stack((angs,mags))

    classifier = GaussianMixture(n_components=NUM_SOURCES)
    classed = classifier.fit_predict(X = toClass)

    for i in range(len(classed-10) //10): # smoothing the classifications
        classed[i*10:i*10 + 10] = np.round(np.mean(classed[i*10:i*10 + 10]))


    indices = np.array(indices)

    #good stuff i think from here on
    sourceindeces = []
    for i in range(NUM_SOURCES):
        sourceindeces.append(indices[classed == i])



    toinverse = []
    for source in sourceindeces:
        temp = np.zeros_like(spec1)
        for idx in source:
            temp[idx[0]][idx[1]] = spec1[idx[0], idx[1]]
        toinverse.append(temp)

    for i in range(NUM_SOURCES):
        toReturn[i,:] = scipy.signal.istft(toinverse[i], fs1, nperseg=nper, noverlap=nover)[1][:len(x1)]

    return 2**15 * toReturn.astype('int16')
















