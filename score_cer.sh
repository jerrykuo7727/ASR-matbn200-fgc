for dset in matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
  steps/scoring/score_kaldi_cer.sh --cmd "run.pl" data/${dset}_hires data/lang_rescore exp/chain_cleaned_1d/tdnn1d_sp/decode_${dset}_rescore

  echo "<WER>"
  for f in  exp/chain_cleaned_1d/tdnn1d_sp/decode_${dset}_rescore/wer*; do echo $f; egrep  '(WER)|(SER)' < $f; done

  echo "<CER>"
for f in  exp/chain_cleaned_1d/tdnn1d_sp/decode_${dset}_rescore/cer*; do echo $f; egrep  '(WER)|(SER)' < $f; done
done
