# Training a reliable speech detector

## What was wrong in the current project

- The research dataset loader did not save its manifest or audio root, so training could not read any examples. It also silently replaced unreadable files with silence.
- The offline spectral fallback turned simple hand-written features into an authenticity verdict. That feature set is not a reliable spoof detector; the backend now abstains when model weights fail to load.
- The research training checkpoint was not connected to backend inference. The backend can now load `best_model_eer.pt` through `ECHOPROOF_CHECKPOINT`.
- The tiny `research/data/asvspoof_sample` files are generated tones for a UI/demo, not ASVspoof speech. The bundled `ground_truth_benchmark` has only 10 short clips. Neither is enough to train or validate a general-purpose detector.
- The old generator split reused all genuine samples in both sets. The split helper now partitions speakers first and checks that an unseen-generator test set remains.

## Data plan

For a strong multilingual detector, combine complementary sources and keep their provenance in the manifest:

1. **[SpeechFake](https://github.com/YMLLG/SpeechFake)** for current TTS, voice conversion, vocoder, and multilingual coverage. The [ACL 2025 paper](https://aclanthology.org/2025.acl-long.493/) reports over 3 million fake samples, more than 3,000 hours, and 46 languages. The released repository describes 30 open-source generation methods; 10 API-generated sets are withheld for licensing reasons. Pair its fake clips with the corresponding permitted genuine source speech, and check every source/model license before redistribution.
2. **[ASVspoof 2019](https://www.asvspoof.org/index2019.html) and [2021](https://www.asvspoof.org/index2021.html) LA/DF** for established synthetic/converted speech and channel/codec shifts. Include **PA** examples when the target is audio captured again through a speaker and microphone; that is a replay condition and differs from detecting a clean generated file.
3. **[MLAAD](https://deepfake-demo.aisec.fraunhofer.de/mlaad)** as a complementary multilingual TTS set. It is spoof-only, so pair it with genuine speech from an allowed source. Its dataset page states a CC BY-NC 4.0 license; that is non-commercial.
4. Add consenting real microphone recordings and captured replays across multiple phones, rooms, distances, and speakers. Keep whole speakers, recording devices, and synthesis systems out of the final test set.

Recent [ACL 2026 work](https://aclanthology.org/2026.acl-long.796/) finds that a diverse, balanced mixture matters more than simply concatenating everything. Training here uses inverse-frequency sampling by `source × generator` as a practical baseline. It is not an implementation of the paper's full DOSS method.

## Manifest format

Prepare separate train and validation CSVs with these columns. Extra columns are allowed.

```csv
filename,label,speaker,generator,source
speech/real_0001.wav,bonafide,spk_001,bonafide,SpeechFake
speech/fake_0001.wav,spoof,spk_001,OpenVoice,SpeechFake
```

`filename` may be an absolute path or a path relative to `--audio_base_dir`. Accepted label names include `bonafide`/`real`/`genuine` and `spoof`/`fake`/`synthetic`. Keep paired genuine/fake speech from the same source on the same side of every split. Do not use the bundled tone demo as training data.

The validation CSV must contain both classes, speakers that are absent from training, and realistic recording/channel conditions. Keep an additional untouched test set with unseen synthesis systems and devices for the final report. Tune decision thresholds on validation only, never on test data.

## Train and connect a checkpoint

The larger multilingual XLS-R backbone is the higher-capacity option and needs a CUDA GPU with ample memory. Reduce batch size or use `facebook/wav2vec2-base` when hardware is limited; that trades capacity for accessibility.

```powershell
python -m pip install -r backend/requirements.txt
python research/train.py `
  --train_csv D:/speech-data/train.csv `
  --val_csv D:/speech-data/validation.csv `
  --audio_base_dir D:/speech-data `
  --output_dir research/results/echoproof_xlsr `
  --backbone_id facebook/wav2vec2-xls-r-300m `
  --freeze_layers 12 `
  --batch_size 2 `
  --epochs 10 `
  --device cuda
```

The trainer selects the checkpoint with the best validation EER, calibrates a genuine/synthetic/uncertain decision band on that validation set, and writes `best_model_eer.pt` plus `decision_thresholds.json`. To use it, set this in the project-root `.env` (use the absolute path on your machine):

```dotenv
ECHOPROOF_CHECKPOINT="E:/path/to/EchoProof AI/research/results/echoproof_xlsr/best_model_eer.pt"
```

Restart the backend after changing `.env`. The `/api/v1/model` endpoint reports which model loaded and whether its thresholds were calibrated. If no neural model loads, analysis now returns **Uncertain** with an explicit warning.

Evaluate the trained checkpoint on the untouched test manifest (with an audio directory containing the referenced files):

```powershell
python research/benchmark.py `
  --model research/results/echoproof_xlsr/best_model_eer.pt `
  --manifest_csv D:/speech-data/test.csv `
  --audio_dir D:/speech-data `
  --output_dir research/results/echoproof_xlsr/test_report `
  --device cuda
```

## Reading results honestly

Report ROC-AUC and EER on the untouched test set, plus false-positive and false-negative rates by language, generator, speaker/device, and clean versus captured audio. State the test sample count and confidence interval. A high score on the bundled ten-clip sample is not evidence of deployment accuracy, and no audio-only detector can guarantee authenticity for every new generator or recording chain.
