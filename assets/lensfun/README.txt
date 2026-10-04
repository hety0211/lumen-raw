Lens profiles for LUMEN RAW's lens corrections (1.5.1)

db/*.xml are the unmodified database files of the lensfun project
(https://lensfun.github.io/, https://github.com/lensfun/lensfun, data/db),
taken from commit bbd4332a9ec566fd9aa548c9e0d8ced238c56261 (2026-09-24).
They are licensed under the Creative Commons Attribution-Share Alike 3.0
license (CC BY-SA 3.0); see COPYING.CC_BY-SA_3.0 in this folder.

LUMEN RAW reads these files and evaluates the published calibration models
(ptlens / poly3 / poly5 distortion, linear / poly3 lateral chromatic
aberration, pa vignetting) in lumen/lens.py; it does not include or link the
lensfun library.  To update the profiles, replace db/*.xml with a newer
lensfun data/db folder and note the commit here.
