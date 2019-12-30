#!/bin/bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# https://lium.univ-lemans.fr/ted-lium3/
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Vincent Nguyen
#            2016  Johns Hopkins University (Author: Daniel Povey)
#            2018  François Hernandez
#
# Apache 2.0
#

. ./cmd.sh
. ./path.sh


set -e -o pipefail -u

nj=35
decode_nj=38   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=16
lm_order=4
train_lm=false
train_rnnlm=false

. utils/parse_options.sh # accept options


# Data preparation
if [ $stage -le 0 ]; then
  echo "========================================"
  echo "    Stage 0 | Prepare data for Kaldi    "
  echo "========================================"

  # Prepare language data
  rm data exp -rf
  mkdir -p data/local/dict 
  echo -e "<sil> sil\n<unk> spn\n`cat $LANG_DIR/lexicon.txt`" > data/local/dict/lexicon.txt
  $python3_cmd scripts/prepare_data.py

  # Combine acoustic data from multiple datasets
  utils/combine_data.sh data/train \
                        $MATBN_DIR/train \
                        $FGC_DIR/grandchallenge-1st-round/train \
                        $FGC_DIR/grandchallenge-2nd-round/train \
                        $FGC_DIR/grandchallenge-3rd-round/train \
                        $FGC_DIR/grandchallenge-4th-round/train \
                        $FGC_DIR/grandchallenge-5th-round/train \
                        $FGC_DIR/grandchallenge-6th-round/train \
                        $FGC_DIR/grandchallenge-final/train \
                        $FGC_DIR/grandchallenge-intermediary-1/train \
                        $FGC_DIR/grandchallenge-intermediary-2/train \
                        $FGC_DIR/grandchallenge-intermediary-3/train \
                        $FGC_DIR/grandchallenge-intermediary-4/train \
                        $FGC_DIR/grandchallenge-mulligan/train

  utils/combine_data.sh data/matbn_dev $MATBN_DIR/dev
  utils/combine_data.sh data/matbn_test $MATBN_DIR/test

  utils/combine_data.sh data/fgc_wo_noise_dev \
                        $FGC_DIR/grandchallenge-1st-round/dev \
                        $FGC_DIR/grandchallenge-2nd-round/dev \
                        $FGC_DIR/grandchallenge-3rd-round/dev \
                        $FGC_DIR/grandchallenge-4th-round/dev \
                        $FGC_DIR/grandchallenge-5th-round/dev
  utils/combine_data.sh data/fgc_wo_noise_test \
                        $FGC_DIR/grandchallenge-1st-round/test \
                        $FGC_DIR/grandchallenge-2nd-round/test \
                        $FGC_DIR/grandchallenge-3rd-round/test \
                        $FGC_DIR/grandchallenge-4th-round/test \
                        $FGC_DIR/grandchallenge-5th-round/test

  utils/combine_data.sh data/fgc_w_noise_dev \
                        $FGC_DIR/grandchallenge-6th-round/dev \
                        $FGC_DIR/grandchallenge-final/dev \
                        $FGC_DIR/grandchallenge-intermediary-1/dev \
                        $FGC_DIR/grandchallenge-intermediary-2/dev \
                        $FGC_DIR/grandchallenge-intermediary-3/dev \
                        $FGC_DIR/grandchallenge-intermediary-4/dev \
                        $FGC_DIR/grandchallenge-mulligan/dev
  utils/combine_data.sh data/fgc_w_noise_test \
                        $FGC_DIR/grandchallenge-6th-round/test \
                        $FGC_DIR/grandchallenge-final/test \
                        $FGC_DIR/grandchallenge-intermediary-1/test \
                        $FGC_DIR/grandchallenge-intermediary-2/test \
                        $FGC_DIR/grandchallenge-intermediary-3/test \
                        $FGC_DIR/grandchallenge-intermediary-4/test \
                        $FGC_DIR/grandchallenge-mulligan/test

  # Prepare rest of necessary data
  for dset in train matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
    utils/utt2spk_to_spk2utt.pl data/$dset/utt2spk > data/$dset/spk2utt
    utils/fix_data_dir.sh data/$dset
    mv data/$dset data/$dset.orig
  done
fi


if [ $stage -le 1 ]; then
  echo "====================================="
  echo "    Stage 1 | Modify speaker info    "
  echo "====================================="

  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
  #  more than our normal 30 jobs.]
  for dset in train matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
  done
fi


if [ $stage -le 2 ]; then
  echo "================================================="
  echo "    Stage 2 | Prepare data for language model    "
  echo "================================================="
  
  echo -n > data/local/dict/extra_questions.txt
  mv data/local/dict data/local/dict_nosp
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
fi


if [ $stage -le 3 ]; then
  echo "================================================="
  echo "    Stage 3 | Train and prune language models    "
  echo "================================================="

  dir=data/local/local_lm
  lm_dir=$dir/data
  
  mkdir -p $dir
  export PATH=$KALDI_ROOT/tools/pocolm/scripts:$PATH
  ( # First make sure the pocolm toolkit is installed.
   cd $KALDI_ROOT/tools || exit 1;
   if [ -d pocolm ]; then
     echo Not installing the pocolm toolkit since it is already there.
   else
     echo "$0: Please install the PocoLM toolkit with: "
     echo " cd ../../../tools; extras/install_pocolm.sh; cd -"
     exit 1;
   fi
  ) || exit 1;


  mkdir -p $dir/data/text
  mkdir -p $dir/data/arpa
  
  # Extract N sentences from tail of corpus as dev set
  num_dev_sentences=10000
  tail -n $num_dev_sentences $LANG_DIR/corpus.txt > $dir/data/text/dev.txt
  head -n -$num_dev_sentences $LANG_DIR/corpus.txt > $dir/data/text/train.txt
  $python3_cmd scripts/extract_text.py data/matbn_dev/text $dir/data/real_dev_set_matbn.txt
  $python3_cmd scripts/extract_text.py data/fgc_wo_noise_dev/text $dir/data/real_dev_set_fgc_wo_noise.txt
  $python3_cmd scripts/extract_text.py data/fgc_w_noise_dev/text $dir/data/real_dev_set_fgc_w_noise.txt

  lm_name=matbn_fgc_cna_"$lm_order"n
  unpruned_lm_dir=$lm_dir/$lm_name.pocolm
  min_counts='train=2'

  train_lm.py  --num-splits=10 --warm-start-ratio=20  \
               --limit-unk-history=true --verbose=true \
               --min-counts="${min_counts}" \
               $dir/data/text $lm_order $lm_dir/work $unpruned_lm_dir

  size=10000000
  prune_lm_dir.py --target-num-ngrams=$size --initial-threshold=0.02 $unpruned_lm_dir $dir/data/lm_${lm_order}_prune_big
  format_arpa_lm.py $dir/data/lm_${lm_order}_prune_big | gzip -c > $dir/data/arpa/${lm_order}gram_big.arpa.gz

  size=2000000
  prune_lm_dir.py --target-num-ngrams=$size $dir/data/lm_${lm_order}_prune_big $dir/data/lm_${lm_order}_prune_small
  format_arpa_lm.py $dir/data/lm_${lm_order}_prune_small | gzip -c > $dir/data/arpa/${lm_order}gram_small.arpa.gz
  
  # Show log probability and perplexity of language models on dev set
  echo "<Perplexity for MATBN-200>"
  get_data_prob.py $dir/data/real_dev_set_matbn.txt $unpruned_lm_dir 2>&1 | grep -F '[perplexity'
  get_data_prob.py $dir/data/real_dev_set_matbn.txt $dir/data/lm_${lm_order}_prune_big 2>&1 | grep -F '[perplexity'
  get_data_prob.py $dir/data/real_dev_set_matbn.txt $dir/data/lm_${lm_order}_prune_small 2>&1 | grep -F '[perplexity'
  echo "<Perplexity for FGC w/o noise>"
  get_data_prob.py $dir/data/real_dev_set_fgc_wo_noise.txt $unpruned_lm_dir 2>&1 | grep -F '[perplexity'
  get_data_prob.py $dir/data/real_dev_set_fgc_wo_noise.txt $dir/data/lm_${lm_order}_prune_big 2>&1 | grep -F '[perplexity'
  get_data_prob.py $dir/data/real_dev_set_fgc_wo_noise.txt $dir/data/lm_${lm_order}_prune_small 2>&1 | grep -F '[perplexity'
  echo "<Perplexity for FGC w/ noise>"
  get_data_prob.py $dir/data/real_dev_set_fgc_w_noise.txt $unpruned_lm_dir 2>&1 | grep -F '[perplexity'
  get_data_prob.py $dir/data/real_dev_set_fgc_w_noise.txt $dir/data/lm_${lm_order}_prune_big 2>&1 | grep -F '[perplexity'
  get_data_prob.py $dir/data/real_dev_set_fgc_w_noise.txt $dir/data/lm_${lm_order}_prune_small 2>&1 | grep -F '[perplexity'
fi


if [ $stage -le 4 ]; then
  echo "============================================================"
  echo "    Stage 4 | Build G.fst / G.carpa with language models    "
  echo "============================================================"

  local/format_lms.sh
fi


# Feature extraction
if [ $stage -le 5 ]; then
  echo "=============================================="
  echo "    Stage 5 | Make MFCC features with CMVN    "
  echo "=============================================="

  for dset in train matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
    dir=data/$dset
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" $dir
    steps/compute_cmvn_stats.sh $dir
  done
fi


# Now we have 452 hours of training data.
# Well create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 6 ]; then
  echo "===================================================="
  echo "    Stage 6 | Create a 10k subset from train set    "
  echo "===================================================="

  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  utils/data/remove_dup_utts.sh 10 data/train_10kshort data/train_10kshort_nodup
fi


# Train
if [ $stage -le 7 ]; then
  echo "================================================="
  echo "    Stage 7 | Monophone | Flat-start training    "
  echo "================================================="

  steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
    data/train_10kshort_nodup data/lang_nosp exp/mono
fi


if [ $stage -le 8 ]; then
  echo "====================================================================="
  echo "    Stage 8 | Triphone (1st pass): Delta + Delta-Delta | Training    "
  echo "====================================================================="

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang_nosp exp/mono_ali exp/tri1
fi


if [ $stage -le 9 ]; then
  echo "====================================================================="
  echo "    Stage 9 | Triphone (1st pass): Delta + Delta-Delta | Decoding    "
  echo "====================================================================="

  utils/mkgraph.sh data/lang_nosp exp/tri1 exp/tri1/graph_nosp

  # The slowest part about this decoding is the scoring, which we can't really
  # control as the bottleneck is the NIST tools.
  for dset in matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri1/graph_nosp data/${dset} exp/tri1/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
       data/${dset} exp/tri1/decode_nosp_${dset} exp/tri1/decode_nosp_${dset}_rescore
  done
fi


if [ $stage -le 10 ]; then
  echo "============================================================="
  echo "    Stage 10 | Triphone (2nd pass): LDA + MLLT | Training    "
  echo "============================================================="

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang_nosp exp/tri1_ali exp/tri2
fi


if [ $stage -le 11 ]; then
  echo "============================================================="
  echo "    Stage 11 | Triphone (2nd pass): LDA + MLLT | Decoding    "
  echo "============================================================="

  utils/mkgraph.sh data/lang_nosp exp/tri2 exp/tri2/graph_nosp
  for dset in matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri2/graph_nosp data/${dset} exp/tri2/decode_nosp_${dset}
    steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
       data/${dset} exp/tri2/decode_nosp_${dset} exp/tri2/decode_nosp_${dset}_rescore
  done
fi


if [ $stage -le 12 ]; then
  echo "=================================================================="
  echo "    Stage 12 | Add pronunciation / silence probability to dict    "
  echo "=================================================================="

  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp exp/tri2
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri2/pron_counts_nowb.txt \
    exp/tri2/sil_counts_nowb.txt \
    exp/tri2/pron_bigram_counts_nowb.txt data/local/dict
fi


if [ $stage -le 13 ]; then
  echo "=================================================================="
  echo "    Stage 13 | Triphone (2nd pass): LDA + MLLT | Decoding w/SP    "
  echo "=================================================================="

  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp -rT data/lang data/lang_rescore
  cp data/lang_nosp/G.fst data/lang/
  cp data/lang_nosp_rescore/G.carpa data/lang_rescore/

  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph

  for dset in matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
    steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri2/graph data/${dset} exp/tri2/decode_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset} exp/tri2/decode_${dset} exp/tri2/decode_${dset}_rescore
  done
fi


if [ $stage -le 14 ]; then
  echo "======================================================================"
  echo "    Stage 14 | Triphone (3rd pass): SAT | Training & Decoding w/SP    "
  echo "======================================================================"

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2_ali exp/tri3

  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

  for dset in matbn_dev matbn_test \
              fgc_wo_noise_dev fgc_wo_noise_test \
              fgc_w_noise_dev fgc_w_noise_test; do
    steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
      exp/tri3/graph data/${dset} exp/tri3/decode_${dset}
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset} exp/tri3/decode_${dset} exp/tri3/decode_${dset}_rescore
  done
fi


if [ $stage -le 15 ]; then
  echo "================================================================================="
  echo "    Stage 15 | Triphone (3rd pass): SAT | Training & Decoding w/SP + data-cleaning    "
  echo "================================================================================="

  # this does some data-cleaning.  It actually degrades the GMM-level results
  # slightly, but the cleaned data should be useful when we add the neural net and chain
  # systems.  If not we'll remove this stage.
  local/run_cleanup_segmentation.sh
fi


if [ $stage -le 16 ]; then
  echo "======================================================================================="
  echo "    Stage 16 | Triphone (4th pass): TDNN | Training & Decoding w/SP + data-cleaning    "
  echo "======================================================================================="

  # This will only work if you have GPUs on your system (and note that it requires
  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
  local/chain/run_tdnn.sh
fi

echo "=============================================="
echo "    'run.sh' ends successfully at stage 16    "
echo "=============================================="
exit 0


if [ $stage -le 18 ]; then
  # You can either train your own rnnlm or download a pre-trained one
  if $train_rnnlm; then
    local/rnnlm/tuning/run_lstm_tdnn_a.sh
    local/rnnlm/average_rnnlm.sh
  else
    local/ted_download_rnnlm.sh
  fi
fi

if [ $stage -le 19 ]; then
  # Here we rescore the lattices generated at stage 17
  rnnlm_dir=exp/rnnlm_lstm_tdnn_a_averaged
  lang_dir=data/lang_chain
  ngram_order=4

  for dset in dev test; do
    data_dir=data/${dset}_hires
    decoding_dir=exp/chain_cleaned/tdnnf_1a/decode_${dset}
    suffix=$(basename $rnnlm_dir)
    output_dir=${decoding_dir}_$suffix

    rnnlm/lmrescore_pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      $lang_dir $rnnlm_dir \
      $data_dir $decoding_dir \
      $output_dir
  done
fi


echo "$0: success."
exit 0
