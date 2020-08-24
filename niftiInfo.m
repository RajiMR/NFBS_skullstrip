%Load preprocessed file
infoPre = niftiinfo('/rsrch1/ip/rmuthusivarajan/imaging/NFBS/nfbsOutput/Test/images/preprocessed_T1w_A00037511.nii');
volPre = niftiread(infoPre);

%Load raw input data
infoRaw = niftiinfo('/rsrch1/ip/rmuthusivarajan/imaging/NFBS/NFBS_Dataset/A00037511/sub-A00037511_ses-NFB3_T1w_brainmask.nii.gz');
volRaw = niftiread(infoRaw);

%write info to preprocessed file from raw file
niftiwrite(volPre,'out.nii',infoRaw);